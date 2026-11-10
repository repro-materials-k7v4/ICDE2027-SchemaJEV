"""Per-dataset original-label metrics, with no model, file or network I/O.

Probabilities and human distributions stay in the caller's original option
order. Neither is renormalized or clipped. Classification gold_index and the
binary Yes index are explicit; scores use the original grid and continuous
human target. Only scalar score pairs and classification confusion counts are
retained, never logits or per-example probability vectors.
"""
import math
import numbers

PROBABILITY_SUM_TOLERANCE = 1e-6

def _need(value, message):
    if not value:
        raise ValueError(message)

def _distribution(values, label):
    result = [float(v) for v in values]
    _need(len(result) >= 2 and all(math.isfinite(v) and 0 <= v <= 1 for v in result), label + ': finite probabilities in [0,1] required')
    _need(abs(math.fsum(result) - 1.0) <= PROBABILITY_SUM_TOLERANCE, label + ': sum must be one within 1e-6; no automatic normalization')
    return result

def _index(value, width, label):
    _need(isinstance(value, numbers.Integral) and not isinstance(value, bool) and 0 <= value < width, label + ': explicit valid record-order integer index required')
    return int(value)

def _correlation(x, y):
    if len(x) < 2:
        return None, 'fewer_than_two_examples'
    mx, my = math.fsum(x) / len(x), math.fsum(y) / len(y)
    dx, dy = [v - mx for v in x], [v - my for v in y]
    sx, sy = max(map(abs, dx)), max(map(abs, dy))
    if sx == 0 or sy == 0:
        reason = 'constant_prediction_and_target' if sx == sy == 0 else ('constant_prediction' if sx == 0 else 'constant_target')
        return None, reason
    # Rescaling the centered pairs avoids overflow of the squared raw scale.
    dx, dy = [v / sx for v in dx], [v / sy for v in dy]
    value = math.fsum(a * b for a, b in zip(dx, dy)) / math.sqrt(math.fsum(v * v for v in dx) * math.fsum(v * v for v in dy))
    return max(-1.0, min(1.0, value)), None

def _average_tie_ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for i in order[start:end]:
            ranks[i] = rank
        start = end
    return ranks

class Accumulator:
    def __init__(self, task_type, candidate_values=None, positive_index=None):
        _need(task_type in ('choice', 'binary', 'score'), 'Explicit task_type choice/binary/score required')
        self.task_type = task_type
        self.candidate_values = None
        self.positive_index = None
        if task_type == 'score':
            _need(candidate_values is not None and positive_index is None, 'Score requires original candidate_values, no binary index')
            grid = [float(v) for v in candidate_values]
            _need(len(grid) >= 2 and len(set(grid)) == len(grid) and all(math.isfinite(v) for v in grid), 'Finite distinct original candidate values in record order required')
            self.candidate_values = tuple(grid)
        elif task_type == 'binary':
            _need(candidate_values is None, 'Binary uses original options, not a numeric score grid')
            self.positive_index = _index(positive_index, 2, 'positive Yes index')
        else:
            _need(candidate_values is None and positive_index is None, 'Choice has neither score grid nor implicit positive label')
        self.count = 0
        self.correct = 0
        self.confusion = []
        self._nll_sum = 0.0
        self._nll_compensation = 0.0
        self._infinite_nll_rows = 0
        self._predictions = []
        self._targets = []

    def add(self, probabilities, human_probability, gold_index=None, original_target=None):
        p = _distribution(probabilities, 'prediction')
        h = _distribution(human_probability, 'original human target')
        _need(len(p) == len(h), 'Prediction/human target original option widths must match')
        if self.task_type == 'score':
            _need(gold_index is None and original_target is not None and len(p) == len(self.candidate_values), 'Score requires original continuous target and exact original grid width, no rounded gold_index')
            target = float(original_target)
            grid = self.candidate_values
            _need(math.isfinite(target) and min(grid) <= target <= max(grid), 'Original target must be finite and within original grid, never clipped')
            expected = math.fsum(v * q for v, q in zip(grid, h))
            tolerance = 1e-6 * max(1.0, abs(min(grid)), abs(max(grid)), max(grid) - min(grid))
            _need(abs(expected - target) <= tolerance, 'Original human soft-grid expectation must preserve continuous target; no rounding or label conversion')
            prediction = math.fsum(v * q for v, q in zip(grid, p))
        else:
            _need(original_target is None, 'Classification takes explicit gold_index, not numeric target conversion')
            if self.task_type == 'binary':
                _need(len(p) == 2, 'Binary must preserve its original two record-order options')
            gold = _index(gold_index, len(p), 'original gold_index')
            predicted = max(range(len(p)), key=p.__getitem__)  # First original option wins an exact tie.
        infinite = any(weight > 0 and probability == 0 for weight, probability in zip(h, p))
        nll = None if infinite else -math.fsum(weight * math.log(probability) for weight, probability in zip(h, p) if weight > 0)
        # Mutate only after every input check, so rejected records cannot count.
        self.count += 1
        if infinite:
            self._infinite_nll_rows += 1
        else:
            value = nll - self._nll_compensation
            updated = self._nll_sum + value
            self._nll_compensation = (updated - self._nll_sum) - value
            self._nll_sum = updated
        if self.task_type == 'score':
            self._predictions.append(prediction)
            self._targets.append(target)
        else:
            width = max(len(p), len(self.confusion))
            for row in self.confusion:
                row.extend([0] * (width - len(row)))
            self.confusion.extend([[0] * width for _ in range(width - len(self.confusion))])
            self.confusion[gold][predicted] += 1
            self.correct += int(gold == predicted)
        return self

    def result(self):
        n = self.count
        nll_reason = 'empty_dataset' if n == 0 else ('positive_human_mass_at_zero_predicted_probability' if self._infinite_nll_rows else None)
        result = {'task_type': self.task_type, 'count': n, 'nll': self._nll_sum / n if n and not self._infinite_nll_rows else None, 'nll_reason': nll_reason, 'nll_infinite_rows': self._infinite_nll_rows, 'probability_renormalization_or_clipping': False}
        if self.task_type == 'score':
            errors = [a - b for a, b in zip(self._predictions, self._targets)]
            mae = math.fsum(map(abs, errors)) / n if n else None
            rmse = math.sqrt(math.fsum(v * v for v in errors) / n) if n else None
            pearson, pearson_reason = _correlation(self._predictions, self._targets)
            spearman, spearman_reason = _correlation(_average_tie_ranks(self._predictions), _average_tie_ranks(self._targets))
            result.update(mae=mae, rmse=rmse, pearson=pearson, pearson_reason=pearson_reason, spearman=spearman, spearman_reason=spearman_reason, candidate_values=list(self.candidate_values), candidate_range=max(self.candidate_values) - min(self.candidate_values), range_normalized_mae=mae / (max(self.candidate_values) - min(self.candidate_values)) if mae is not None else None, original_scale_metrics=True, score_target_rounding_or_clipping=False)
            return result
        matrix = [row[:] for row in self.confusion]
        if not matrix and self.task_type == 'binary':
            matrix = [[0, 0], [0, 0]]
        support = [sum(row) for row in matrix]
        predicted_support = [sum(row[i] for row in matrix) for i in range(len(matrix))]
        result.update(accuracy=self.correct / n if n else None, accuracy_reason=None if n else 'empty_dataset', correct=self.correct, confusion_matrix=matrix, confusion_order='rows=original_gold_index,columns=original_predicted_index', gold_support=support, predicted_support=predicted_support)
        if self.task_type == 'binary':
            pos, neg = self.positive_index, 1 - self.positive_index
            tp, fn, fp, tn = matrix[pos][pos], matrix[pos][neg], matrix[neg][pos], matrix[neg][neg]
            positive_support, negative_support, predicted_positive = tp + fn, tn + fp, tp + fp
            # With no actual positive examples the requested positive-class
            # metrics are explicitly undefined, even if false positives exist.
            precision_reason = 'zero_positive_support' if not positive_support else ('zero_predicted_positive' if not predicted_positive else None)
            recall_reason = 'zero_positive_support' if not positive_support else None
            f1_reason = 'zero_positive_support' if not positive_support else None
            balanced_reason = 'zero_positive_support' if not positive_support else ('zero_negative_support' if not negative_support else None)
            result.update(positive_index=pos, positive_label='Yes', positive_support=positive_support, negative_support=negative_support, predicted_positive=predicted_positive, true_positive=tp, false_positive=fp, false_negative=fn, true_negative=tn, precision=tp / predicted_positive if precision_reason is None else None, precision_reason=precision_reason, recall=tp / positive_support if recall_reason is None else None, recall_reason=recall_reason, f1=2 * tp / (2 * tp + fp + fn) if f1_reason is None else None, f1_reason=f1_reason, balanced_accuracy=(tp / positive_support + tn / negative_support) / 2 if balanced_reason is None else None, balanced_accuracy_reason=balanced_reason)
        return result

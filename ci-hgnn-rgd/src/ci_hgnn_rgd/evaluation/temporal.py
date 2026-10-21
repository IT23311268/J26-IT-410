"""Chronological split-window checks; actual dataset allocation is deferred."""

from pydantic import AwareDatetime, model_validator

from ..schemas import FrozenModel


class TemporalWindow(FrozenModel):
    """Half-open interval [start, end)."""

    start: AwareDatetime
    end: AwareDatetime

    @model_validator(mode="after")
    def ordered(self):
        if self.start >= self.end:
            raise ValueError("window start must precede end")
        return self


def validate_split_windows(train: TemporalWindow, validation: TemporalWindow, test: TemporalWindow):
    if train.end > validation.start or validation.end > test.start:
        raise ValueError("train, validation, and test windows must be chronological and disjoint")

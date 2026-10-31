"""Standard policy checkpoints use their own explicit action schema."""
from .rustmage_policy import RustMagePolicy
from .standard_features import StandardEncoder


class StandardPolicy(RustMagePolicy):
    def __init__(self, model, encoder=None, **kwargs):
        super().__init__(model, encoder or StandardEncoder(), **kwargs)

    @classmethod
    def load(cls, path, *, encoder=None, **options):
        return super().load(path, encoder=encoder or StandardEncoder(), **options)

"""Standard policy checkpoints use their own explicit action schema."""
from .rustmage_policy import RustMagePolicy
from .standard_features import StandardEncoder
from . import compact_standard_model as registered_compact_models  # noqa: F401


def model_encoder(model):
    if getattr(model,'input_format',None)=='sparse-identity-standard-v2':
        from .compact_identity import IdentityEncoder
        return IdentityEncoder()
    return StandardEncoder()


class StandardPolicy(RustMagePolicy):
    def __init__(self, model, encoder=None, **kwargs):
        super().__init__(model, encoder or model_encoder(model), **kwargs)

    @classmethod
    def load(cls, path, *, encoder=None, **options):
        import json
        from pathlib import Path
        saved=json.loads(Path(path).with_suffix('.manifest.json').read_text())
        if saved.get('representation')=='sparse-identity-standard-v2':
            from . import compact_identity_model as registered_identity_models
            from .compact_identity import IdentityEncoder
            encoder=encoder or IdentityEncoder()
        return super().load(path, encoder=encoder or StandardEncoder(), **options)

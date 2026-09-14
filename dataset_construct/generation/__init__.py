"""Generation module: candidate generation, transformation, ambiguity injection, spec generation, and feature generation."""
from .candidates import get_key_feature_generator, KeyFeatureGenerator
from .transform import get_transform_processor, TransformProcessor, TransformResult
from .ambiguity import get_ambiguity_injector, AmbiguityInjector, AmbiguityDiversityConstraints
from .spec import get_spec_generator, SpecGenerator
from .features import get_required_key_feature_generator, RequiredKeyFeatureGenerator

__all__ = [
    "get_key_feature_generator", "KeyFeatureGenerator",
    "get_transform_processor", "TransformProcessor", "TransformResult",
    "get_ambiguity_injector", "AmbiguityInjector", "AmbiguityDiversityConstraints",
    "get_spec_generator", "SpecGenerator",
    "get_required_key_feature_generator", "RequiredKeyFeatureGenerator",
]

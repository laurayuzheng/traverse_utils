from unitraj.datasets.MTR_dataset import MTRDataset
from unitraj.datasets.autobot_dataset import AutoBotDataset
from unitraj.datasets.wayformer_dataset import WayformerDataset

from .balance_dataset import BalancedDataset
from .persona_dataset import PersonaDataset

__all__ = {
    'autobot': AutoBotDataset,
    'wayformer': WayformerDataset,
    'MTR': PersonaDataset,
    'MTR_context': MTRDataset, 
    'MTR_LoRA': MTRDataset,
    'PC_MTR_LoRA': PersonaDataset,
    'Discrete_PC_MTR_LoRA': PersonaDataset,
} # type: ignore


def build_dataset(config, val=False):
    requires_balancing = getattr(config, 'oversampling', False) or getattr(config, 'undersampling', False)
    if not val and requires_balancing:
        return BalancedDataset(config=config, is_validation=val)

    dataset = __all__[config.method.model_name](
        config=config, is_validation=val
    )
    return dataset

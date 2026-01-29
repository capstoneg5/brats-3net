from dataclasses import dataclass

@dataclass(frozen=True)
class ModelConfig:
    name: str = "unet3d"
    # ----------------------
    # Segmentation
    # ----------------------
    IMG_SIZE = (128, 128, 128)
    SEGMENTATION_IN_CHANNELS = 4
    SEGMENTATION_OUT_CHANNELS = 4

    # ----------------------
    # Embeddings
    # ----------------------
    TEXT_EMBEDDING_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
    VISION_EMBEDDING_MODEL = "microsoft/swin-base-patch4-window7-224"
    MULTIMODAL_EMBEDDING_MODEL = "openai/clip-vit-base-patch32"

    EMBEDDING_DIM = 768


# singleton instance used everywhere
model_config = ModelConfig()
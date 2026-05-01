"""
utils.py  –  Utility functions (e.g. selectively unfreeze ViT encoder blocks).
"""
def unfreeze_last_n_vit_blocks(vit_model, n_blocks):
    total_blocks = len(vit_model.vit.encoder.layer)

    # Freeze all first
    for block in vit_model.vit.encoder.layer:
        block.trainable = False

    # Unfreeze last n blocks
    for block in vit_model.vit.encoder.layer[-n_blocks:]:
        block.trainable = True

    print(f"✅ Unfroze last {n_blocks}/{total_blocks} ViT encoder blocks")

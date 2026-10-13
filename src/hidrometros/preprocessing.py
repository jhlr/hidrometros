from pathlib import Path
from PIL import Image

def make_digit_crops(strip: Image.Image, expected_size=(384, 64)):
    """
    Recebe uma faixa de leitura já orientada/corrigida e recortada.
    Espera tamanho (largura, altura) = (384, 64) e devolve 8 recortes
    de 48x64, da esquerda para a direita.

    Atenção: esta função NÃO localiza automaticamente a faixa no hidrômetro,
    não corrige perspectiva e não determina a orientação. Essas etapas dependem
    das anotações e do protocolo real do dataset.
    """
    strip = strip.convert("RGB")
    if strip.size != expected_size:
        raise ValueError(
            f"Tamanho esperado {expected_size} (LxA), recebido {strip.size}. "
            "Redimensione/recorte conforme o protocolo validado."
        )
    width, height = strip.size
    crop_width = width // 8
    crops = []
    for i in range(8):
        left = i * crop_width
        right = (i + 1) * crop_width
        crops.append(strip.crop((left, 0, right, height)))
    return crops

def save_crops(crops, output_dir, stem="reading"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, crop in enumerate(crops):
        path = output_dir / f"{stem}_digit_{i}.png"
        crop.save(path)
        paths.append(path)
    return paths

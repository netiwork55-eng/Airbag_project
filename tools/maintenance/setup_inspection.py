"""Install approved inspection dependencies without modifying the existing camera environment."""
import argparse
import importlib.metadata
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--fetch-backbone',action='store_true',help='Download the official TorchVision ResNet18 V1 weights if absent')
    parser.add_argument('--fetch-detectors',action='store_true',help='Cache official TorchVision COCO V1 weights for detection and segmentation training')
    args=parser.parse_args()
    if sys.version_info[:2]!=(3,12):
        raise SystemExit('This tested installation uses Python 3.12. Validate other Python versions separately.')
    expected={'torch':'2.13.0','torchvision':'0.28.0','numpy':'2.5.1'}
    for name,version in expected.items():
        if importlib.metadata.version(name)!=version:
            raise SystemExit(f'{name} must match the tested base runtime {version}; do not silently upgrade it')
    env=ROOT/'.venv-inspection'
    python=env/'Scripts/python.exe'
    if not python.exists():
        subprocess.run([sys.executable,'-m','venv',str(env)],check=True)
    base=Path(sys.prefix)/'Lib/site-packages'
    if not base.is_dir():
        raise SystemExit('Run this setup with the existing Windows project virtual environment')
    (env/'Lib/site-packages/existing_runtime.pth').write_text(str(base.resolve())+'\n',encoding='utf-8')
    subprocess.run([str(python),'-m','pip','install','--index-url','https://pypi.org/simple',
                    '-r',str(ROOT/'requirements-inspection.lock')],check=True)
    subprocess.run([str(python),'-m','pip','check'],check=True)
    if args.fetch_backbone:
        # Official weights URL and built-in hash validation, not an arbitrary model download.
        from torchvision.models import ResNet18_Weights
        ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=True,check_hash=True)
    if args.fetch_detectors:
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, MaskRCNN_ResNet50_FPN_Weights
        for weights in (FasterRCNN_ResNet50_FPN_Weights.COCO_V1, MaskRCNN_ResNet50_FPN_Weights.COCO_V1):
            weights.get_state_dict(progress=True,check_hash=True)
    print('Inspection worker ready. This has not activated or changed any Production recipe.')


if __name__=='__main__':main()

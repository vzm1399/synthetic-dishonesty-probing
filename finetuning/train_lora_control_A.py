import sys
from train_lora import main


if __name__ == '__main__':
    if '--label' not in sys.argv:
        sys.argv.extend(['--label', 'honest'])
    main()

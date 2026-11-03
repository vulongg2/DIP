import sys
import os

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))
    from scripts.run_multi_cam import main
    main()

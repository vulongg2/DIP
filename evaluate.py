import sys
import os

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))
    from scripts.evaluate_benchmark import main
    main()

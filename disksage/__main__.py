import sys

from .cli import main

if __name__ == "__main__":
    # main() loads user config (~/.config/disksage/.env) and any local ./.env.
    sys.exit(main())

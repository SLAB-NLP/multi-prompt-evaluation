import argparse
import sys
from pathlib import Path

# Add project root to Python path
file_path = Path(__file__).parents[2]
sys.path.append(str(file_path))

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--server_port', default=None, type=str,
                       help='Port to run the app on. Example: --server_port 8501')
    args = parser.parse_args()

    # Configure Streamlit CLI arguments
    sys.argv = ["streamlit", "run", "app.py",
                "--server.address", "localhost",
                "--server.enableCORS", "false",
                "--server.enableXsrfProtection", "false",
                "--global.suppressDeprecationWarnings", "true",
                "--client.showErrorDetails", "false",
                "--client.toolbarMode", "minimal"]

    # Add port if specified
    if args.server_port:
        sys.argv.extend(["--server.port", args.server_port])

    # Run the app
    from streamlit.web import cli as stcli
    sys.exit(stcli.main())
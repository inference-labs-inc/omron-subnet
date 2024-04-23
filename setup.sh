#!/bin/bash

# Throw on error
set -e
# Grab OS ID
OS="`uname`"
# Flag to let the user know if they need to restart their terminal
SHOULD_RESTART=false

# OS specific python install
install_python() {
    case $OS in
      'Linux')
        sudo apt update
        sudo apt install -y python3 python3-pip
        SHOULD_RESTART=true
        ;;
      'Darwin')
        brew install python
        SHOULD_RESTART=true
        ;;
      *)
        echo "Unsupported OS"
        exit 1
        ;;
    esac
}

# OS specific node install
install_node() {
    case $OS in
      'Linux')
        sudo apt update
        sudo apt install -y nodejs npm
        SHOULD_RESTART=true
        ;;
      'Darwin')
        brew install node
        SHOULD_RESTART=true
        ;;
      *)
        echo "Unsupported OS"
        exit 1
        ;;
    esac
}

# Check if brew is installed, if not then install it
if [[ "$OS" == "Darwin" && -z "$(command -v brew)" ]]; then
  echo "brew not found. Installing brew..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Check if Python3 is installed, if not then install it
if [ -z "$(command -v python3)" ]; then
  echo "Python3 not found. Installing Python3..."
  install_python
fi

# Check if Node is installed, if not then install it
if [ -z "$(command -v node)" ]; then
  echo "Node not found. Installing Node..."
  install_node
fi

# Check if btcli is installed, if not then install it
if [ -z "$(command -v btcli)" ]; then
  echo "btcli not found. Installing btcli..."
  python3 -m pip install bittensor
  SHOULD_RESTART=true
fi

# Check if PM2 is installed, if not then install it
if [ -z "$(command -v pm2)" ]; then
  echo "pm2 not found. Installing pm2..."
  sudo npm install -g pm2
fi



# Ask user where they want to install the SN
read -p "Where would you like to install Omron? (./omron): " INSTALL_PATH </dev/tty
INSTALL_PATH=${INSTALL_PATH:-./omron}

# Clone SN repo into user's specified directory
git clone https://github.com/inference-labs-inc/omron-subnet.git $INSTALL_PATH

# Setup a Virtual Python environment for dependencies
if ! python3 -m venv --help > /dev/null 2>&1; then
  echo "venv module not found. Installing venv..."
  sudo apt-get install -y python3-venv
fi
echo "Setting up Python virtual environment..."
python3 -m venv $INSTALL_PATH/.venv
source $INSTALL_PATH/.venv/bin/activate

# Install Python dependencies from requirements.txt
echo "Installing Python dependencies..."
python3 -m pip install -r $INSTALL_PATH/requirements.txt

# Show completion message and prompt user to restart their terminal if necessary
if [ "$SHOULD_RESTART" = true ]; then
  echo -e "\033[32mInstallation complete. Please restart your terminal for the changes to take effect.\033[0m"
else
  echo -e "\033[32mInstallation complete. \033[0m"
fi

# Download the SN's PK
echo "Downloading PK..."
sudo wget https://storage.omron.ai/pk_xs.key -O $INSTALL_PATH/neurons/deployment_layer/model_0/pk.key

echo "Downloading SRS..."
sudo wget https://storage.omron.ai/kzg_xs.srs -O $INSTALL_PATH/neurons/deployment_layer/model_0/kzg.srs

# Display next steps
echo -e "\033[32mOmron has been installed to ${INSTALL_PATH}. Please run \`cd ${INSTALL_PATH}\` to navigate to the directory.\033[0m"
echo -e "\033[32mPlease see ${INSTALL_PATH}/docs/shared_setup_steps.md for the next steps.\033[0m"

#!/bin/bash

# Throw on error
set -e
# Grab OS ID
OS="$(uname)"
# Flag to let the user know if they need to restart their terminal
SHOULD_RESTART=false

# OS specific python install
install_python() {
	case ${OS} in
	'Linux')
		# FIXME(Ivan): Linux is not always Ubuntu, checking DISTRIB_ID is recommended here
		sudo apt update
		sudo apt install -y python3 python3-pip python3-venv
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
	case ${OS} in
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
if [[ ${OS} == "Darwin" ]] && ! command -v brew >/dev/null 2>&1; then
	echo "brew not found. Installing brew..."
	/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" || true
fi

# Check if Python3 is installed, if not then install it
if ! command -v python3 >/dev/null 2>&1 || ! command -v pip3 >/dev/null 2>&1; then
	echo "Python3 not found. Installing Python3..."
	install_python
fi

if ! python3 -m pip --version >/dev/null 2>&1; then
	echo "pip module not found. Installing..."
	install_python
fi

if ! dpkg -s python3-venv >/dev/null 2>&1; then
	echo "python3-venv package not found. Installing..."
	sudo apt install -y python3-venv
fi

# Check if Node and npm are installed, if not then install them
if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
	echo "Node or npm not found. Installing Node and npm..."
	install_node
fi

# Check if jq is installed, if not then install it
if ! command -v jq >/dev/null 2>&1; then
	echo "jq not found. Installing jq..."
	case ${OS} in
	'Linux')
		sudo apt update
		sudo apt install -y jq
		;;
	'Darwin')
		brew install jq
		;;
	*)
		echo "Unsupported OS for jq installation"
		exit 1
		;;
	esac
fi

# Check if PM2 is installed, if not then install it
if ! command -v pm2 >/dev/null 2>&1; then
	echo "pm2 not found. Installing pm2..."
	sudo npm install -g pm2
fi
# Ask user where they want to install the SN
read -rp "Where would you like to install Omron? (./omron): " INSTALL_PATH </dev/tty
INSTALL_PATH=${INSTALL_PATH:-./omron}

# Clone SN repo into user's specified directory
if ! [[ -d ${INSTALL_PATH} ]]; then
	git clone https://github.com/inference-labs-inc/omron-subnet.git "${INSTALL_PATH}"
else
	echo "${INSTALL_PATH} already exists"
fi

# Ask user for virtualenv type
read -rp "Which virtualenv would you like to use? (venv/virtualenv/none): " VENV_TYPE </dev/tty
VENV_TYPE=${VENV_TYPE:-none}

# Install and activate virtualenv if requested
if [[ ${VENV_TYPE} == "venv" ]]; then
	echo "Setting up venv..."
	python3 -m venv "${INSTALL_PATH}"/omron-venv
	source "${INSTALL_PATH}"/omron-venv/bin/activate
elif [[ ${VENV_TYPE} == "virtualenv" ]]; then
	echo "Setting up virtualenv..."
	sudo pip3 install virtualenv
	virtualenv "${INSTALL_PATH}"/omron-venv
	source "${INSTALL_PATH}"/omron-venv/bin/activate
fi

# Install Python dependencies from requirements.txt
echo "Installing Python dependencies..."
python3 -m pip install -r "${INSTALL_PATH}"/requirements.txt

# Check if btcli is installed, if not then install it
if ! command -v btcli >/dev/null 2>&1; then
	echo "btcli not found. Installing btcli...!!!"
	python3 -m pip install bittensor
	SHOULD_RESTART=true
fi

# Show completion message and prompt user to restart their terminal if necessary
if [[ ${SHOULD_RESTART} == true ]]; then
	echo -e "\033[32mInstallation complete. Please restart your terminal for the changes to take effect.\033[0m"
else
	echo -e "\033[32mInstallation complete. \033[0m"
fi

# Set working directory to install dir
cd "${INSTALL_PATH}" || exit

# Sync remote files for all models
echo "Syncing model files..."
bash "./sync_model_files.sh"

# Display next steps
echo -e "\033[32mOmron has been installed to ${INSTALL_PATH}. Please run \`cd ${INSTALL_PATH}\` to navigate to the directory.\033[0m"
echo -e "\033[32mPlease see ${INSTALL_PATH}/docs/shared_setup_steps.md for the next steps.\033[0m"

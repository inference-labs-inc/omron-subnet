#!/bin/bash

# Throw on error
set -e
# Grab OS ID
OS="$(uname)"
# Flag to let the user know if they need to restart their terminal
SHOULD_RESTART=false
# Flag to check if we should only install dependencies
NO_INSTALL=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
	case $1 in
	--no-install) NO_INSTALL=true ;;
	*)
		echo "Unknown parameter passed: $1"
		exit 1
		;;
	esac
	shift
done

# OS specific python install
install_python() {
	case ${OS} in
	'Linux')
		# FIXME(Ivan): Linux is not always Ubuntu, checking DISTRIB_ID is recommended here
		sudo apt update
		sudo apt install -y software-properties-common
		sudo add-apt-repository -y ppa:deadsnakes/ppa
		sudo apt update
		sudo apt install -y python3.10 python3.10-venv python3-pip
		sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
		sudo update-alternatives --set python3 /usr/bin/python3.10
		SHOULD_RESTART=true
		;;
	'Darwin')
		brew update
		brew install python@3.10
		brew link python@3.10
		SHOULD_RESTART=true
		;;
	*)
		echo "Unsupported OS"
		exit 1
		;;
	esac
}

set_python310_default() {
	case ${OS} in
	'Linux')
		sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
		sudo update-alternatives --set python3 /usr/bin/python3.10
		sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
		sudo update-alternatives --set python /usr/bin/python3.10
		;;
	'Darwin')
		brew link --force python@3.10
		echo 'export PATH="/usr/local/opt/python@3.10/bin:$PATH"' >>~/.bash_profile
		echo 'export PATH="/usr/local/opt/python@3.10/bin:$PATH"' >>~/.zshrc
		;;
	*)
		echo "Unsupported OS for setting Python 3.10 as default"
		return 1
		;;
	esac
	SHOULD_RESTART=true
}
# OS specific node and npm install
install_node() {
	NODE_VERSION="20.16.0"

	case ${OS} in
	'Linux')
		# Remove conflicting packages
		sudo apt-get remove -y nodejs npm
		sudo apt-get autoremove -y
		sudo apt-get remove -y libnode-dev
		# Install Node.js and npm
		echo "Installing Node.js ${NODE_VERSION} and npm..."
		curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
		sudo apt-get install -y nodejs
		;;
	'Darwin')
		echo "Installing Node.js ${NODE_VERSION} and npm..."
		brew update
		brew install node@20
		brew link --force node@20
		;;
	*)
		echo "Unsupported OS"
		exit 1
		;;
	esac

	# Verify installation
	if node --version | grep -q "${NODE_VERSION}" && npm --version >/dev/null 2>&1; then
		echo "Node.js version ${NODE_VERSION} and npm are available."
	else
		echo "Failed to verify Node.js and npm installation."
		exit 1
	fi

	SHOULD_RESTART=true
}

# Check if brew is installed, if not then install it
if [[ ${OS} == "Darwin" ]] && ! command -v brew >/dev/null 2>&1; then
	echo "brew not found. Installing brew..."
	/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" || true
fi

# Check if Python3 is installed and version is 3.10, if not then install it
if ! command -v python3 >/dev/null 2>&1 || ! command -v pip3 >/dev/null 2>&1 || ! python3 -c "import sys; assert sys.version_info[:2] == (3, 10), 'Python 3.10 required'" >/dev/null 2>&1; then
	echo "Python 3.10 not found. Installing Python 3.10..."
	install_python
fi

if ! python3 -m pip --version >/dev/null 2>&1; then
	echo "pip module not found. Installing..."
	install_python
fi

if python3 -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 10) else 1)" >/dev/null 2>&1; then
	echo "Python 3.10 is installed. Setting it as default..."
	set_python310_default
else
	echo "Python 3.10 is not installed. Please check your installation."
	exit 1
fi

if ! dpkg -s python3-venv >/dev/null 2>&1; then
	echo "python3-venv package not found. Installing..."
	sudo apt install -y python3-venv
fi

# Check if Node.js version 20 and npm are installed, if not then install them
if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1 || ! node --version | grep -q "^v20\."; then
	echo "Node.js version 20 or npm not found. Installing Node.js version 20 and npm..."
	install_node
else
	echo "Node.js version 20 and npm are already installed."
fi

# Check if snarkjs is installed, if not then install it
# Define the local installation directory
local_snarkjs_dir="${HOME}/.snarkjs"
local_snarkjs_path="${local_snarkjs_dir}/node_modules/.bin/snarkjs"

if ! command -v "${local_snarkjs_path} r1cs info --help" >/dev/null 2>&1; then
	echo "SnarkJS 0.7.4 not found in local directory. Installing..."
	mkdir -p "${local_snarkjs_dir}"
	npm install --prefix "${local_snarkjs_dir}" snarkjs@0.7.4
	echo "SnarkJS has been installed in the local directory."
fi

# Add the local snarkjs directory to the PATH
export PATH="$(dirname "${local_snarkjs_path}"):${PATH}"

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

# If NO_INSTALL is true, exit here
if [ "$NO_INSTALL" = true ]; then
	echo "Dependencies checked and installed. Exiting without installing Omron."
	exit 0
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

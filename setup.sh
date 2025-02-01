#!/bin/bash

set -eo pipefail


PYTHON_VERSION="3.10"
NODE_VERSION="20"
INSTALL_PATH="./omron"


BREW_PACKAGES=(
    "python@${PYTHON_VERSION}"
    "node@${NODE_VERSION}"
    "jq"
    "aria2"
    "pkg-config"
    "openssl"
)

APT_PACKAGES=(
    "jq"
    "aria2"
    "pkg-config"
    "libssl-dev"
    "openssl"
    "python${PYTHON_VERSION}"
    "python${PYTHON_VERSION}-venv"
    "python3-pip"
)

NPM_PACKAGES=(
    "pm2"
    "snarkjs@0.7.4"
)


case "$(uname)" in
    "Darwin")
        if ! command -v brew &>/dev/null; then
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            eval "$(/opt/homebrew/bin/brew shellenv)"
        fi

        echo "Installing brew packages..."
        brew update
        for pkg in "${BREW_PACKAGES[@]}"; do
            brew install "$pkg" || brew upgrade "$pkg"
        done

        brew link --force "python@${PYTHON_VERSION}"
        brew link --force "node@${NODE_VERSION}"

        npm config set cafile /etc/ssl/cert.pem

        if [ -d "$HOME/.npm" ]; then
            sudo chown -R $(whoami):$(id -g) "$HOME/.npm"
        fi
        ;;

    "Linux")
        echo "Installing apt packages..."
        sudo apt update
        sudo apt install -y software-properties-common
        sudo add-apt-repository -y ppa:deadsnakes/ppa
        sudo apt update
        sudo apt install -y "${APT_PACKAGES[@]}"


        sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1
        sudo update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION}


        curl -fsSL "https://deb.nodesource.com/setup_${NODE_VERSION}.x" | sudo -E bash -
        sudo apt install -y nodejs
        ;;

    *)
        echo "Unsupported OS"
        exit 1
        ;;
esac

local_snarkjs_dir="${HOME}/.snarkjs"
local_snarkjs_path="${local_snarkjs_dir}/node_modules/.bin/snarkjs"

echo "Installing npm packages..."
for pkg in "${NPM_PACKAGES[@]}"; do
    if [[ "$pkg" == "snarkjs@0.7.4" ]]; then
        if ! command -v "${local_snarkjs_path} r1cs info --help" >/dev/null 2>&1; then
            echo "SnarkJS 0.7.4 not found in local directory. Installing..."
            mkdir -p "${local_snarkjs_dir}"
            npm install --prefix "${local_snarkjs_dir}" snarkjs@0.7.4
            echo "SnarkJS has been installed in the local directory."
        fi
    else
        sudo npm install -g "$pkg"
    fi
done

curl -LsSf https://astral.sh/uv/install.sh | sh

export PATH="$HOME/.local/bin:$PATH"
chmod +x "$HOME/.local/bin/uv"
chmod +x "$HOME/.local/bin/uvx"

if [[ ! -d ${INSTALL_PATH} ]]; then
    echo "Cloning omron-subnet repository..."
    if ! git clone https://github.com/inference-labs-inc/omron-subnet.git "${INSTALL_PATH}"; then
        echo "Failed to clone repository. Check your internet connection and try again."
        exit 1
    fi
fi

cd "${INSTALL_PATH}" || {
    echo "Failed to change to ${INSTALL_PATH} directory"
    exit 1
}

"$HOME/.local/bin/uv" venv
source "${INSTALL_PATH}/.venv/bin/activate"


"$HOME/.local/bin/uv" sync --locked

echo "
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@%#%@@@@@@@@@@@@@@@@@@
@@@@@@@@*.......*@@@@@@@@@@@@@@@
@@@@@@@+.........+@@@@@@@@@@@@@@
@@@@@@@:.....-#%%+..:=%@@@@@@@@@
@@@@@@@*...:#@@@@=.....=@@@@@@@@
@@@@@@@@#-.#@@@#-.......+@@@@@@@
@@@@@@@@@@*.-:..........:@@@@@@@
@@@@@@@@@@%.............-@@@@@@@
@@@@@@@@@@@*...........:%@@@@@@@
@@@@@@@@@@@@@-........*@@@@@@@@@
@@@@@@@@@@@@@@@%###%%@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
 ██████╗ ███╗   ███╗██████╗  ██████╗ ███╗   ██╗
██╔═══██╗████╗ ████║██╔══██╗██╔═══██╗████╗  ██║
██║   ██║██╔████╔██║██████╔╝██║   ██║██╔██╗ ██║
██║   ██║██║╚██╔╝██║██╔══██╗██║   ██║██║╚██╗██║
╚██████╔╝██║ ╚═╝ ██║██║  ██║╚██████╔╝██║ ╚████║
 ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
███████╗██╗   ██╗██████╗ ███╗   ██╗███████╗████████╗
██╔════╝██║   ██║██╔══██╗████╗  ██║██╔════╝╚══██╔══╝
███████╗██║   ██║██████╔╝██╔██╗ ██║█████╗     ██║
╚════██║██║   ██║██╔══██╗██║╚██╗██║██╔══╝     ██║
███████║╚██████╔╝██████╔╝██║ ╚████║███████╗   ██║
╚══════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝   ╚═╝
"
echo "🥩 Setup complete! Next steps:"
echo "1. cd ${INSTALL_PATH}"
echo "2. source venv/bin/activate"
echo "3. Check docs/shared_setup_steps.md"

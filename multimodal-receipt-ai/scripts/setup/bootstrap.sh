#!/bin/bash
# scripts/bootstrap.sh
# Installs 'task' (go-task) if missing.

OS="$(uname -s)"
ARCH="$(uname -m)"

if command -v task &> /dev/null; then
    echo "✅ task is already installed."
    task --version
    exit 0
fi

echo "⚠️  task not found. Attempting installation..."

if [[ "$OS" == "Darwin" ]]; then
    if command -v brew &> /dev/null; then
        echo "🍺 Installing via Homebrew..."
        brew install go-task/tap/go-task
    else
        echo "❌ Homebrew not found. Falling back to install script."
        sh -c "$(curl --location https://taskfile.dev/install.sh)" -- -d -b /usr/local/bin
    fi
elif [[ "$OS" == "Linux" ]]; then
    # Try snap? Or raw install
    sh -c "$(curl --location https://taskfile.dev/install.sh)" -- -d -b ~/.local/bin
    export PATH="$PATH:$HOME/.local/bin"
    echo "Please ensure ~/.local/bin is in your PATH"
else
    echo "❌ Unsupported OS: $OS"
    exit 1
fi

if command -v task &> /dev/null; then
    echo "✅ task installed successfully!"
else
    # Check if it was installed to /usr/local/bin but not in path for some reason
    if [ -f "/usr/local/bin/task" ]; then
         echo "✅ task installed to /usr/local/bin/task"
    elif [ -f "$HOME/.local/bin/task" ]; then
         echo "✅ task installed to ~/.local/bin/task"
    else
         echo "❌ Installation failed."
         exit 1
    fi
fi

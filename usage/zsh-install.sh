# 检查 zsh 是否已安装
echo "检查 zsh 是否已安装..."
if ! command -v zsh &> /dev/null; then
    echo "zsh 未安装，正在安装..."
    # 对于 Debian/Ubuntu 系统
    sudo apt update && sudo apt install -y curl zsh unzip git
    # 对于 CentOS/RHEL 系统
    # sudo yum install -y zsh unzip git
    # 对于 macOS 用户，可以通过 Homebrew 安装
    # brew install zsh unzip git
fi

# 切换默认 shell 为 zsh
echo "切换默认 shell 为 zsh..."
chsh -s $(which zsh)

# 安装 Oh My Zsh
echo "检查 Oh My Zsh 是否已安装..."
if [ ! -d "$HOME/.oh-my-zsh" ]; then
    echo "正在安装 Oh My Zsh..."
    yes | sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
else
    echo "Oh My Zsh 已经安装。"
fi

# 安装 Dracula 主题
echo "正在安装 Dracula 主题..."
wget -O dracula.zip -c --no-check-certificate https://github.com/dracula/zsh/archive/master.zip
unzip dracula.zip
mv zsh-master/dracula.zsh-theme ~/.oh-my-zsh/themes/
mv zsh-master/lib ~/.oh-my-zsh/themes/lib
rm -rf dracula.zip zsh-master

# 创建 ~/.zshrc 文件并写入内容
cat << 'EOF' > ~/.zshrc
# Path to your Oh My Zsh installation.
export ZSH="$HOME/.oh-my-zsh"

# Set name of the theme to load
ZSH_THEME="dracula"

# Which plugins would you like to load?
plugins=(
    git
    autojump 
    zsh-autosuggestions
    zsh-syntax-highlighting
)

# User configuration
export UPDATE_ZSH_DAYS=30 # 30天检查更新zsh
HIST_STAMPS="yyyy-mm-dd"  # 历史记录时间格式
export LANG=en_US.UTF-8   # 语言

[[ -s ~/.autojump/etc/profile.d/autojump.sh ]] && . ~/.autojump/etc/profile.d/autojump.sh

source $ZSH/oh-my-zsh.sh
EOF

# 安装 zsh-autosuggestions 插件
echo "正在安装 zsh-autosuggestions 插件..."
git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions

# 安装 autojump 插件
echo "正在安装 autojump 插件..."
git clone https://github.com/joelthelion/autojump.git
cd autojump
./install.py
cd ..
rm -rf autojump

# 安装 zsh-syntax-highlighting 插件
echo "正在安装 zsh-syntax-highlighting 插件..."
git clone https://github.com/zsh-users/zsh-syntax-highlighting ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting

# 提示用户重启终端或切换到 zsh
echo "安装完成，请重启终端或运行 'source ~/.zshrc' 以应用更改。"
exec zsh  # 自动切换到 zsh

set encoding=utf-8
set fileencodings=utf-8,chinese,latin-1
if has("win32")
    set fileencoding=chinese
else
    set fileencoding=utf-8
endif

set autoindent
set shiftwidth=4
syntax enable 
syntax on
set showmatch
set smartindent
set shiftwidth=4
set ai!
set guicursor+=a:blinkon0
set nobackup
set ts=4
set expandtab
set ruler
set nu!


set guioptions-=T
set guioptions-=m

source $VIMRUNTIME/delmenu.vim
source $VIMRUNTIME/menu.vim
source $VIMRUNTIME/vimrc_example.vim
source $VIMRUNTIME/mswin.vim
behave mswin

language messages zh_CN.utf-8

colorscheme molokai

set guifont=Courier_new:h15:b:cDEFAULT
set statusline=[%F]%y%r%m%*%=[Line:%l/%L,Column:%c][%p%%]
set fdm=marker
set number
set cursorline

nmap <F6> :w!<CR>:!python %<CR>

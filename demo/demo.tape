# Records demo/demo.gif with VHS (brew install vhs).
#   bash demo/make_demo_home.sh && vhs demo/demo.tape
# It runs against the fake home from make_demo_home.sh, so no real paths appear.
# Record after installing disksage (pipx install disksage).

Output demo/demo.gif

Set Shell "bash"
Set FontSize 16
Set Width 1280
Set Height 760
Set Padding 20
Set TypingSpeed 45ms

Hide
Type "export HOME=/tmp/disksage-demo && cd $HOME && clear"
Enter
Show

Type "disksage models"
Sleep 400ms
Enter
Sleep 4s

Type "disksage scan --no-ai"
Sleep 400ms
Enter
Sleep 7s

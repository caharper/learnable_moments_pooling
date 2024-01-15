Replace <repo_name>

From your computer to superpod (in directory above aperture-layers):

`rsync -arvhp --exclude='*.git' --filter="dir-merge,- ./<repo_name>/.superpod_ignore" ./<repo_name> caharper@superpod.smu.edu:/users/caharper`


From superpod to your computer (in directory above <repo_name>):

`rsync -arvhp --exclude='*.git' --filter="dir-merge,- ./<repo_name>.superpod_ignore" caharper@superpod.smu.edu:/users/caharper ./`
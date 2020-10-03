# append the project root path to bashrc
printf "\nexport PYTHONPATH=$PYTHONPATH:$PWD" >> ~/.bashrc
source ~/.bashrc

echo "Setup completed. You might have to run 'source ~/.bashrc'"
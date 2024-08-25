# Go to parent directory, where `requirements.txt` is
cd $PROJECT_DIR

# Print everything into the file
pip freeze > requirements.txt

# Log changes
printf "Successfully updated requirements.txt!\n"

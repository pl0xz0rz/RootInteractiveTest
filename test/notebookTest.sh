for note in $(ls */*/*ipynb); do	for note in $(ls */*/*ipynb); do
    echo jupyter nbconvert ${note}	    echo mv ${note}2 ${note}
    jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace ${note}	    jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace ${note}

done	done
py.test --nbval  $(ls */*/*ipynb | grep -v 460) | tee notebookTest.log

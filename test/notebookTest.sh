for note in $(ls */*/*ipynb); do
    echo jupyter nbconvert ${note}
    jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace ${note}
done
py.test --nbval  $(ls */*/*ipynb | grep -v 460 |grep -v 566|grep -v 8275) | tee notebookTest.log

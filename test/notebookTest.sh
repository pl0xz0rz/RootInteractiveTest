#clean_ipynb */*/*ipynb
for note in $(ls */*/*ipynb); do
    echo mv ${note}2 ${note}
    jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace ${note}

done

py.test --nbval  $(ls */*/*ipynb | grep -v 460) | tee pytest.log


# py.test --nbval JIRA/ATO-457/distortionConvergenceStudy.ipynb - needed to fix osbolte bokehDrawPanda

#clean_ipynb */*/*ipynb
for note in $(ls */*/*ipynb); do
    nb-clean clean < ${note}  > ${note}2
    echo mv ${note}2 ${note}
    mv ${note}2 ${note}
done

py.test --nbval  $(ls */*/*ipynb | grep -v 460) | tee pytest.log


# py.test --nbval JIRA/ATO-457/distortionConvergenceStudy.ipynb - needed to fix osbolte bokehDrawPanda

py.test --nbval $(ls */*ipynb | grep -v 460) | tee pytest.log 

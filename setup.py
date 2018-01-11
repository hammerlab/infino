from setuptools import setup

setup(name='infino',
      version='0.1',
      description='infer immune infiltrate phenotypes',
      url='http://github.com/hammerlab/infino',
      author='Hammer Lab',
      author_email='correspondence@hammerlab.org',
      packages=['infino'],
      install_requires=[
          'numpy',
          'matplotlib',
          'pandas',
          'pystan',
          'seaborn',
      ],
      entry_points = {
        'console_scripts': [
        	'chunker=infino.chunker:main', 
        	'execute-model=infino.execute_model:main',
        	'run-stansummary=infino.run_stansummary:main'
        ],
      }
      zip_safe=False)
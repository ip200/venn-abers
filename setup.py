from distutils.core import setup
setup(
  name='venn-abers',
  packages=['src/'],
  version='0.1',
  license='MIT',
  description='Venn-ABERS calibration package',
  author='Ivan Petej',
  author_email='ivan.petej@gmail.com',
  url='https://github.com/ip200/venn-abers',
  download_url='https://github.com/ip200/venn-abers/archive/refs/tags/v_01.tar.gz',
  keywords=['Probabilistic classification', 'calibration'],
  install_requires=[
          'numpy',
          'scikit-learn',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Machine learning',
    'License :: MIT License',
    'Programming Language :: Python :: 3.11',
  ],
)
from distutils.core import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
  name='venn-abers',
  packages=['venn_abers'],
  package_dir={'venn_abers': 'src'},
  version='1.3',
  license='MIT',
  description='Venn-ABERS calibration package',
  long_description=long_description,
  long_description_content_type="text/markdown",
  author='Ivan Petej',
  author_email='ivan.petej@gmail.com',
  url='https://github.com/ip200/venn-abers',
  download_url='https://github.com/ip200/venn-abers/archive/refs/tags/v1_3.tar.gz',
  keywords=['Probabilistic classification', 'calibration'],
  install_requires=[
          'numpy',
          'scikit-learn',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    "License :: OSI Approved :: MIT License",
    'Programming Language :: Python :: 3',
  ],
)

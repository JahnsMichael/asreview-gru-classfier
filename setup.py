from setuptools import setup
from setuptools import find_namespace_packages

setup(
    name='asreview-lstm-gru-classifier',
    version='1.0',
    description='LSTM and GRU Classifier extension for ASReview',
    url='https://github.com/asreview/asreview',
    author='Jahns Michael',
    author_email='michael.jf.jm@gmail.com',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='systematic review',
    packages=find_namespace_packages(include=['asreviewcontrib.*']),
    python_requires='~=3.6',
    install_requires=[
        'asreview>=1.0',
        'tensorflow'
    ],
    entry_points={
        'asreview.models.classifiers': [
            'gru = asreviewcontrib.models:GatedRecurrentUnit',
            'lstm = asreviewcontrib.models:LSTM',
        ],
        'asreview.models.feature_extraction': [
            # define feature_extraction algorithms
        ],
        'asreview.models.balance': [
            # define balance strategy algorithms
        ],
        'asreview.models.query': [
            # define query strategy algorithms
        ]
    },
    project_urls={
        'Bug Reports': 'https://github.com/asreview/asreview/issues',
        'Source': 'https://github.com/asreview/asreview/',
    },
)

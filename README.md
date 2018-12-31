# IMDB GAN

## Introduction

This project aims at creating a Generative Adversarial Network capable to create convincing
fake reviews on IMDB. This is not to be used to flood the website but is just a proof of concept
that GAN can be used to generate text and not only pictures through convolutional neural networks (CNN) towers
like they are usually.

Furthermore, this is still extremely WIP and not suited for any useful application, but might one day be.

For now, the project's folders are not all populated since there is no need for unit testing a project that is not
yet even functional.

## Project Layout

* cache: Preprocessed datasets that don’t need to be re-generated every time you perform an analysis.
* config: Configuration settings for the project
* data: Raw data files.
* preprocessing: Preprocessing data munging *scripts*, the outputs of which are put in cache.
* src: Statistical analysis scripts.
* diagnostics: Scripts to diagnose data sets for corruption or outliers.
* doc: Documentation written about the analysis.
* graphs: Graphs created from analysis.
* lib: Helper library functions but not the core statistical analysis.
* logs: Output of scripts and any automatic logging.
* profiling: Scripts to benchmark the timing of your code.
* reports: Output reports and content that might go into reports such as tables.
* tests: Unit tests and regression suite for your code.
* notebooks: Jupyter/IPython notebooks to test out code before implementation 
* README.md: Notes that orient any newcomers to the project.

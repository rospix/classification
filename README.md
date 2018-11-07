# Rospix Classification

A ROS node for basic morphologic track classification.

![rospix](misc/rospix.png)

## Dependencies

To download the repository install *git* with *lfs*:
```bash
sudo apt install git git-lfs
```

To run the Python code, following python packages are required:
* **scikit-image** for image feature extraction
* **sklearn** 0.20.0 for machine learning and classification
```
sudo pip3 install scikit-image sklearn
```

## References
1. T. Baca, D. Turecek, R. McEntaffer and R. Filgas, *Rospix: Modular Software Tool for Automated Data Acquisitions of Timepix Detectors on Robot Operating System*, Journal of Instrumentation, 2018.
2. M. Jilek, **[Processing of Radiation Data from the Timepix Sensor on the VZLUSAT-1 Satellite](https://dspace.cvut.cz/bitstream/handle/10467/77036/F3-DP-2018-Jilek-Martin-thesis.pdf)**, Master's thesis, Czech Technical University in Prague, Faculty of Electrical Engineering, 2018.

# Acknowledgements

The work has been done on behalf of Medipix2 collaboration and is currently supported by the Czech Science Foundation project 18-10088Y and by Czech Technical University grant no. SGS17/187/OHK3/3T/13.

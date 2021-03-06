.. :changelog:
2.3.0 (2015-03-11)
~~~~~~~~~~~~~~~~~~
- add get_true_anomaly() method
- remove redundant arrays
- improve accuracy for special cases (e.g. rp near 0.5)

2.2.0 (2015-12-19)
~~~~~~~~~~~~~~~~~~
- add inverse transit capability (can now handle negative rp)
- speed up super-sampling


2.1.0 (2015-08-06)
~~~~~~~~~~~~~~~~~~
- add get_t_conjunction() method 
- change eclipse model normalization so that stellar flux is unity

2.0.0 (2015-08-04)
~~~~~~~~~~~~~~~~~~
- add secondary eclipse model
- change model parameterization to time of inferior conjunction from time of periastron (backwards-incompatible change in response to referee)


1.0.0 (2015-07-29)
~~~~~~~~~~~~~~~~~~
- first stable release


0.9.1 (2015-06-24)
~~~~~~~~~~~~~~~~~~

- fixing bug in call to _rsky


0.9.0 (2015-06-24)
~~~~~~~~~~~~~~~~~~

- Beta version 

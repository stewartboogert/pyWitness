import pytest

def test_05_fitting_test1_csv_indep_obs_eqvar():
    import pyWitness
    dr = pyWitness.DataRaw("../data/tutorial/test1.csv")
    dr.collapseContinuousData(column="confidence", bins=[-1, 60, 80, 100], labels=[1, 2, 3])
    dp = dr.process()
    mf = pyWitness.ModelFitIndependentObservation(dp)
    mf.setEqualVariance()
    mf.fit()

    assert mf.chi2 == pytest.approx(10.300411274463412, rel=1e-5)
    assert mf.numberDegreesOfFreedom == 4
    assert mf.chi2PerNDF == pytest.approx(2.575102818615853, rel=1e-5)
    assert mf.pValue == pytest.approx(0.03566019782522267, rel=1e-5)
    assert mf.lureMean.value == pytest.approx(0.0, rel=1e-5)
    assert mf.lureSigma.value == pytest.approx(1.0, rel=1e-5)
    assert mf.targetMean.value == pytest.approx(1.7976601843420954, rel=1e-5)
    assert mf.targetSigma.value == pytest.approx(1.0, rel=1e-5)
    assert mf.lureBetweenSigma.value == pytest.approx(0.6046983921244553, rel=1e-5)
    assert mf.targetBetweenSigma.value == pytest.approx(0.6046983921244553, rel=1e-5)
    assert mf.c1.value == pytest.approx(1.4017022884785224, rel=1e-5)
    assert mf.c2.value == pytest.approx(1.93548009449426, rel=1e-5)
    assert mf.c3.value == pytest.approx(2.677475327674742, rel=1e-5)

def test_05_fitting_test1_csv_indep_obs_uneqvar():
    import pyWitness
    dr = pyWitness.DataRaw("../data/tutorial/test1.csv")
    dr.collapseContinuousData(column="confidence", bins=[-1, 60, 80, 100], labels=[1, 2, 3])
    dp = dr.process()
    mf = pyWitness.ModelFitIndependentObservation(dp)
    mf.setUnequalVariance()
    mf.fit()

    assert mf.chi2 == pytest.approx(4.534824840657988, rel=1e-5)
    assert mf.numberDegreesOfFreedom == 3
    assert mf.chi2PerNDF == pytest.approx(1.5116082802193294, rel=1e-5)
    assert mf.pValue == pytest.approx(0.20920494600886963, rel=1e-5)
    assert mf.lureMean.value == pytest.approx(0.0, rel=1e-5)
    assert mf.lureSigma.value == pytest.approx(1.0, rel=1e-5)
    assert mf.targetMean.value == pytest.approx(1.8894645402811472, rel=1e-5)
    assert mf.targetSigma.value == pytest.approx(0.7886365473806328, rel=1e-5)
    assert mf.lureBetweenSigma.value == pytest.approx(0.4480311026689192, rel=1e-5)
    assert mf.targetBetweenSigma.value == pytest.approx(0.4480311026689192, rel=1e-5)
    assert mf.c1.value == pytest.approx(1.5351065745707397, rel=1e-5)
    assert mf.c2.value == pytest.approx(2.002341182989407, rel=1e-5)
    assert mf.c3.value == pytest.approx(2.6308524212497644, rel=1e-5)

def test_05_fitting_test1_csv_indep_obs_free():
    import pyWitness
    dr = pyWitness.DataRaw("../data/tutorial/test1.csv")
    dr.collapseContinuousData(column="confidence", bins=[-1, 40, 60, 80, 100], labels=[1, 2, 3, 4])
    dp = dr.process()
    mf = pyWitness.ModelFitIndependentObservation(dp)
    mf.fit()

    assert mf.chi2 == pytest.approx(10.76520159182259, rel=1e-5)
    assert mf.numberDegreesOfFreedom == 2
    assert mf.chi2PerNDF == pytest.approx(5.382600795911295, rel=1e-5)
    assert mf.pValue == pytest.approx(0.004595853496030644, rel=1e-5)
    assert mf.lureMean.value == pytest.approx(0.0006818129624015098, rel=1e-5)
    assert mf.lureSigma.value == pytest.approx(0.7613182180813651, rel=1e-5)
    assert mf.targetMean.value == pytest.approx(1.493581552198572, rel=1e-5)
    assert mf.targetSigma.value == pytest.approx(0.5701995798268649, rel=1e-5)
    assert mf.lureBetweenSigma.value == pytest.approx(-0.0012213835415482996, rel=1e-5)
    assert mf.targetBetweenSigma.value == pytest.approx(0.0012586054967293737, rel=1e-5)
    assert mf.c1.value == pytest.approx(1.25713233870699, rel=1e-5)
    assert mf.c2.value == pytest.approx(1.3660364365503272, rel=1e-5)
    assert mf.c3.value == pytest.approx(1.5870619583040122, rel=1e-5)
    assert mf.c4.value == pytest.approx(2.033939924498857, rel=1e-5)

def test_05_fitting_test1_csv_best_rest_eqvar():
    import pyWitness
    dr = pyWitness.DataRaw("../data/tutorial/test1.csv")
    dr.collapseContinuousData(column="confidence")
    dp = dr.process()
    mf = pyWitness.ModelFitBestRest(dp)
    mf.setEqualVariance()
    mf.fit()

    assert mf.chi2 == pytest.approx(23.203557131914106, rel=1e-5)
    assert mf.numberDegreesOfFreedom == 4
    assert mf.chi2PerNDF == pytest.approx(5.800889282978527, rel=1e-5)
    assert mf.pValue == pytest.approx(0.00011530375016544081, rel=1e-5)
    assert mf.lureMean.value == pytest.approx(0.0, rel=1e-5)
    assert mf.lureSigma.value == pytest.approx(1.0, rel=1e-5)
    assert mf.targetMean.value == pytest.approx(2.03566192879208, rel=1e-5)
    assert mf.targetSigma.value == pytest.approx(1.0, rel=1e-5)
    assert mf.lureBetweenSigma.value == pytest.approx(-0.027650790308206018, rel=1e-5)
    assert mf.targetBetweenSigma.value == pytest.approx(-0.027650790308206018, rel=1e-5)
    assert mf.c1.value == pytest.approx(1.7353510716101122, rel=1e-5)
    assert mf.c2.value == pytest.approx(2.212736570519106, rel=1e-5)
    assert mf.c3.value == pytest.approx(2.9491721140631633, rel=1e-5)

def test_05_fitting_test1_csv_best_rest_uneqvar():
    import pyWitness
    dr = pyWitness.DataRaw("../data/tutorial/test1.csv")
    dr.collapseContinuousData(column="confidence")
    dp = dr.process()
    mf = pyWitness.ModelFitBestRest(dp)
    mf.setUnequalVariance()
    mf.fit()

    assert mf.chi2 == pytest.approx(7.868684226378687, rel=1e-5)
    assert mf.numberDegreesOfFreedom == 3
    assert mf.chi2PerNDF == pytest.approx(2.622894742126229, rel=1e-5)
    assert mf.pValue == pytest.approx(0.04880501500208234, rel=1e-5)
    assert mf.lureMean.value == pytest.approx(0.0, rel=1e-5)
    assert mf.lureSigma.value == pytest.approx(1.0, rel=1e-5)
    assert mf.targetMean.value == pytest.approx(2.038036936738798, rel=1e-5)
    assert mf.targetSigma.value == pytest.approx(0.6139740611429738, rel=1e-5)
    assert mf.lureBetweenSigma.value == pytest.approx(0.03895753863353904, rel=1e-5)
    assert mf.targetBetweenSigma.value == pytest.approx(0.03895753863353904, rel=1e-5)
    assert mf.c1.value == pytest.approx(1.7546932765997934, rel=1e-5)
    assert mf.c2.value == pytest.approx(2.168399339678346, rel=1e-5)
    assert mf.c3.value == pytest.approx(2.7606531061053246, rel=1e-5)

def test_05_fitting_test1_csv_best_rest_free():
    import pyWitness
    dr = pyWitness.DataRaw("../data/tutorial/test1.csv")
    dr.collapseContinuousData(column="confidence", bins=[-1, 40, 60, 80, 100], labels=[1, 2, 3, 4])
    dp = dr.process()
    mf = pyWitness.ModelFitBestRest(dp)
    mf.fit()

    assert mf.chi2 == pytest.approx(9.92916150379401, rel=1e-5)
    assert mf.numberDegreesOfFreedom == 2
    assert mf.chi2PerNDF == pytest.approx(4.964580751897005, rel=1e-5)
    assert mf.pValue == pytest.approx(0.006980876815037118, rel=1e-5)
    assert mf.lureMean.value == pytest.approx(0.0018987441970468675, rel=1e-5)
    assert mf.lureSigma.value == pytest.approx(0.6861322853955167, rel=1e-5)
    assert mf.targetMean.value == pytest.approx(1.4012317220792951, rel=1e-5)
    assert mf.targetSigma.value == pytest.approx(0.4268722771827042, rel=1e-5)
    assert mf.lureBetweenSigma.value == pytest.approx(0.00035201600917691706, rel=1e-5)
    assert mf.targetBetweenSigma.value == pytest.approx(0.0008327066039283845, rel=1e-5)
    assert mf.c1.value == pytest.approx(1.2028664376190874, rel=1e-5)
    assert mf.c2.value == pytest.approx(1.29788822134375, rel=1e-5)
    assert mf.c3.value == pytest.approx(1.4905841431052758, rel=1e-5)

def test_05_fitting_test1_csv_ensemble_eqvar():
    import pyWitness
    dr = pyWitness.DataRaw("../data/tutorial/test1.csv")
    dr.collapseContinuousData(column="confidence")
    dp = dr.process()
    mf = pyWitness.ModelFitEnsemble(dp)
    mf.setEqualVariance()
    mf.fit()

    assert mf.chi2 == pytest.approx(23.20355799888823, rel=1e-5)
    assert mf.numberDegreesOfFreedom == 4
    assert mf.chi2PerNDF == pytest.approx(5.800889499722057, rel=1e-5)
    assert mf.pValue == pytest.approx(0.00011530370414902791, rel=1e-5)
    assert mf.lureMean.value == pytest.approx(0.0, rel=1e-5)
    assert mf.lureSigma.value == pytest.approx(1.0, rel=1e-5)
    assert mf.targetMean.value == pytest.approx(2.03564527876075, rel=1e-5)
    assert mf.targetSigma.value == pytest.approx(1.0, rel=1e-5)
    assert mf.lureBetweenSigma.value == pytest.approx(0.0025862757713492144, rel=1e-5)
    assert mf.targetBetweenSigma.value == pytest.approx(0.0025862757713492144, rel=1e-5)
    assert mf.c1.value == pytest.approx(1.4461421641934356, rel=1e-5)
    assert mf.c2.value == pytest.approx(1.8439293779396513, rel=1e-5)
    assert mf.c3.value == pytest.approx(2.457635059181422, rel=1e-5)

def test_05_fitting_test1_csv_ensemble_uneqvar():
    import pyWitness
    dr = pyWitness.DataRaw("../data/tutorial/test1.csv")
    dr.collapseContinuousData(column="confidence")
    dp = dr.process()
    mf = pyWitness.ModelFitEnsemble(dp)
    mf.setUnequalVariance()
    mf.fit()

    assert mf.chi2 == pytest.approx(7.868684371812639, rel=1e-5)
    assert mf.numberDegreesOfFreedom == 2
    assert mf.chi2PerNDF == pytest.approx(2.6228947906042133, rel=1e-5)
    assert mf.pValue == pytest.approx(0.04880501181888286, rel=1e-5)
    assert mf.lureMean.value == pytest.approx(0.0, rel=1e-5)
    assert mf.lureSigma.value == pytest.approx(1.0, rel=1e-5)
    assert mf.targetMean.value == pytest.approx(2.0380474980579812, rel=1e-5)
    assert mf.targetSigma.value == pytest.approx(0.6139499862139856, rel=1e-5)
    assert mf.lureBetweenSigma.value == pytest.approx(-0.045359525568252, rel=1e-5)
    assert mf.targetBetweenSigma.value == pytest.approx(-0.045359525568252, rel=1e-5)
    assert mf.c1.value == pytest.approx(1.4622645372725116, rel=1e-5)
    assert mf.c2.value == pytest.approx(1.807018272986906, rel=1e-5)
    assert mf.c3.value == pytest.approx(2.3005632098784905, rel=1e-5)

def test_05_fitting_test1_csv_integration():
    import pyWitness
    dr = pyWitness.DataRaw("../data/tutorial/test1.csv")
    dr.collapseContinuousData(column="confidence")
    dp = dr.process()
    mf_in = pyWitness.ModelFitIntegration(dp)

def test_05_fitting_test1_csv_set_parameters_plot_hit_v_false():
    import pyWitness
    dr = pyWitness.DataRaw("../data/tutorial/test1.csv")
    dr.collapseContinuousData(column="confidence", bins=[-1, 60, 80, 100], labels=[1, 2, 3])
    dp = dr.process()
    dp.plotHitVsFalseAlarmRate()

def test_05_fitting_test1_csv_set_parameters_print_parameters():
    import pyWitness
    dr = pyWitness.DataRaw("../data/tutorial/test1.csv")
    dr.collapseContinuousData(column="confidence", bins=[-1, 60, 80, 100], labels=[1, 2, 3])
    dp = dr.process()
    mf = pyWitness.ModelFitIndependentObservation(dp)
    mf.printParameters()

def test_05_fitting_test1_csv_set_parameters_set_equal_var():
    import pyWitness
    dr = pyWitness.DataRaw("../data/tutorial/test1.csv")
    dr.collapseContinuousData(column="confidence", bins=[-1, 60, 80, 100], labels=[1, 2, 3])
    dp = dr.process()
    mf = pyWitness.ModelFitIndependentObservation(dp)
    mf.setEqualVariance()
    mf.setParameterEstimates()
    mf.printParameters()

def test_05_fitting_test1_csv_plot_fit_roc():
    import pyWitness
    dr = pyWitness.DataRaw("../data/tutorial/test1.csv")
    dr.collapseContinuousData(column="confidence", bins=[-1, 60, 80, 100], labels=None)
    dp = dr.process()
    dp.calculateConfidenceBootstrap(nBootstraps=200)
    mf = pyWitness.ModelFitIndependentObservation(dp)
    mf.setEqualVariance()
    mf.fit()
    dp.plotROC(label="Data")
    mf.plotROC(label="Indep. obs. fit")

def test_05_fitting_test1_csv_plot_fit():
    import pyWitness
    dr = pyWitness.DataRaw("../data/tutorial/test1.csv")
    dr.collapseContinuousData(column="confidence", bins=[-1, 60, 80, 100], labels=None)
    dp = dr.process()
    dp.calculateConfidenceBootstrap(nBootstraps=200)
    mf = pyWitness.ModelFitIndependentObservation(dp)
    mf.setEqualVariance()
    mf.fit()
    dp.plotROC(label="Data")
    mf.plotROC(label="Indep. obs. fit")
    mf.plotFit()

def test_05_fitting_test1_csv_plot_fit_cac():
    import pyWitness
    dr = pyWitness.DataRaw("../data/tutorial/test1.csv")
    dr.collapseContinuousData(column="confidence", bins=[-1, 60, 80, 100], labels=None)
    dp = dr.process()
    dp.calculateConfidenceBootstrap(nBootstraps=200)
    mf = pyWitness.ModelFitIndependentObservation(dp)
    mf.setEqualVariance()
    mf.fit()
    dp.plotCAC(label="Data")
    mf.plotCAC(label="Indep. obs. fit")
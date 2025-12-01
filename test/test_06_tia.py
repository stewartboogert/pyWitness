import pandas as pd
import numpy as np
import pyWitness
import pytest

def test_05_fitting_test4_xlsx_ensemble_taluredistribution() :
    # load raw data 
    dr = pyWitness.DataRaw("../data/tutorial/test4.xlsx", excelSheet='raw')

    # collapse/bin data
    dr.collapseContinuousData(column="confidence", bins=[-1, 30.5, 50.5, 70.5, 90.5, 100.5], labels=[1, 2, 3, 4, 5])

    # process 
    dpL = dr.process(column="group", condition="l")

    # make fits 
    mfL = pyWitness.ModelFitLureTargetPresentEnsemble(dpL)
    mfL.setUnequalVariance()

    # set values
    mfL.innocentMean.value = 0
    mfL.innocentSigma.value = 1
    mfL.lureTPMean.value = 0
    mfL.lureMean.value = 0

    # which parameters to fix
    mfL.lureMean.fixed = True
    mfL.lureSigma.fixed = False
    mfL.lureBetweenSigma.fixed = True
    mfL.targetBetweenSigma.fixed = True
    mfL.lureTPMean.fixed = True
    mfL.innocentMean.fixed = True
    mfL.innocentSigma.fixed = True

    # do the fit 
    mfL.fit()

    assert mfL.lureMean.value == 0.000
    assert mfL.lureSigma.value == pytest.approx(1.8087301166363934, rel=1e-5)
    assert mfL.targetMean.value == pytest.approx(3.412627794176202, rel=1e-5)
    assert mfL.targetSigma.value == pytest.approx(1.1965797879678846, rel=1e-5)
    assert mfL.lureBetweenSigma.value == 0.1
    assert mfL.c1.value == pytest.approx(2.3585851280256924, rel=1e-5)
    assert mfL.c2.value == pytest.approx(2.519471176193736, rel=1e-5)
    assert mfL.c3.value == pytest.approx(2.8136524705862214, rel=1e-5)
    assert mfL.c4.value == pytest.approx(3.443072646490758, rel=1e-5)
    assert mfL.c5.value == pytest.approx(4.526363698925529, rel=1e-5)
    assert mfL.innocentMean.value == 0.000
    assert mfL.innocentSigma.value == 1.0
    assert mfL.lureTPMean.value == 0.000
    assert mfL.lureTPSigma.value == pytest.approx(1.4904634644839219, rel=1e-5)
    
def test_05_fitting_test4_xlsx_ensemble_compositefits() :
    # load raw data 
    dr = pyWitness.DataRaw("../data/tutorial/test4.xlsx", excelSheet='raw')

    # collapse/bin data
    dr.collapseContinuousData(column="confidence", bins=[-1, 30.5, 50.5, 70.5, 90.5, 100.5], labels=[1, 2, 3, 4, 5])

    # process 
    dpL = dr.process(column="group", condition="l")
    dpH = dr.process(column="group", condition="h")


    # make fits 
    mfL = pyWitness.ModelFitLureTargetPresentEnsemble(dpL)
    mfL.setUnequalVariance()
    mfH = pyWitness.ModelFitLureTargetPresentEnsemble(dpH)
    mfH.setUnequalVariance()

    # set values
    mfL.innocentMean.value = 0
    mfL.innocentSigma.value = 1
    mfL.lureTPMean.value = 0
    mfL.lureMean.value = 0
    mfH.innocentMean.value = 0
    mfH.innocentSigma.value = 1
    mfH.lureTPMean.value = 0
    mfH.lureMean.value = 0

    # which parameters to fix
    mfL.lureMean.fixed = True
    mfL.lureSigma.fixed = False
    mfL.lureBetweenSigma.fixed = True
    mfL.targetBetweenSigma.fixed = True
    mfL.lureTPMean.fixed = True
    mfL.innocentMean.fixed = True
    mfL.innocentSigma.fixed = True
    mfH.lureMean.fixed = True
    mfH.lureSigma.fixed = False
    mfH.lureBetweenSigma.fixed = True
    mfH.targetBetweenSigma.fixed = True
    mfH.lureTPMean.fixed = True
    mfH.innocentMean.fixed = True
    mfH.innocentSigma.fixed = True

    #couple targetSigma
    mfL.targetSigma = mfH.targetSigma

    #create composite model
    mfc = pyWitness.ModelFitComposite()
    mfc.addFit(mfL)
    mfc.addFit(mfH)
    
    # do the fit 
    mfc.fit(maxiter=15000)
    
    #print 
    mfc.model_fits[0].printParameters()    
    mfc.model_fits[1].printParameters()    
    

    assert mfc.model_fits[0].lureMean.value == 0.000
    assert mfc.model_fits[0].lureSigma.value == pytest.approx(1.926324116518119, rel=1e-5)
    assert mfc.model_fits[0].targetMean.value == pytest.approx(3.625197584309032, rel=1e-5)
    assert mfc.model_fits[0].targetSigma.value == pytest.approx(1.3380788482604513, rel=1e-5)
    assert mfc.model_fits[0].lureBetweenSigma.value == 0.1
    assert mfc.model_fits[0].c1.value == pytest.approx(2.500809738182628, rel=1e-5)
    assert mfc.model_fits[0].c2.value == pytest.approx(2.673131607945537, rel=1e-5)
    assert mfc.model_fits[0].c3.value == pytest.approx(2.990093663163473, rel=1e-5)
    assert mfc.model_fits[0].c4.value == pytest.approx(3.6687375894587833, rel=1e-5)
    assert mfc.model_fits[0].c5.value == pytest.approx(4.852329183139265, rel=1e-5)
    assert mfc.model_fits[0].innocentMean.value == 0.000
    assert mfc.model_fits[0].innocentSigma.value == 1.0
    assert mfc.model_fits[0].lureTPMean.value == 0.000
    assert mfc.model_fits[0].lureTPSigma.value == pytest.approx(1.5879987608447377, rel=1e-5)

    assert mfc.model_fits[1].lureMean.value == 0.000
    assert mfc.model_fits[1].lureSigma.value == pytest.approx(1.6921434649217288, rel=1e-5)
    assert mfc.model_fits[1].targetMean.value == pytest.approx(2.3351771464253637, rel=1e-5)
    assert mfc.model_fits[1].targetSigma.value == pytest.approx(1.3380788482604513, rel=1e-4)
    assert mfc.model_fits[1].lureBetweenSigma.value == 0.1
    assert mfc.model_fits[1].c1.value == pytest.approx(2.154319972040857, rel=1e-5)
    assert mfc.model_fits[1].c2.value == pytest.approx(2.379010226884601, rel=1e-5)
    assert mfc.model_fits[1].c3.value == pytest.approx(2.7173941664896732, rel=1e-5)
    assert mfc.model_fits[1].c4.value == pytest.approx(3.37635171215614, rel=1e-5)
    assert mfc.model_fits[1].c5.value == pytest.approx(4.447843140352736, rel=1e-5)
    assert mfc.model_fits[1].innocentMean.value == 0.000
    assert mfc.model_fits[1].innocentSigma.value == 1.0
    assert mfc.model_fits[1].lureTPMean.value == 0.000
    assert mfc.model_fits[1].lureTPSigma.value == pytest.approx(1.7971739571383303, rel=1e-5)


def test_06_adv_stats_test5_xlsx_pairedroc() :
    #load and process data 
    drHigh = pyWitness.DataRaw("../data/tutorial/test5.xlsx","raw")
    nH = drHigh.cutData("group","h")
    drHigh.collapseContinuousData(column="confidence",bins=[-1,30.5,50.5,70.5,90.5,100.5], labels=[1,2,3,4,5])
    dpHigh = drHigh.process()
    
    drLow = pyWitness.DataRaw("../data/tutorial//test5.xlsx","raw")
    nL = drLow.cutData("group","l")
    drLow.collapseContinuousData(column="confidence",bins=[-1,30.5,50.5,70.5,90.5,100.5], labels=[1,2,3,4,5])
    dpLow = drLow.process()

    #calculate minimum FAR and process
    minRate = min(dpHigh.liberalTargetAbsentSuspectId,dpLow.liberalTargetAbsentSuspectId)

    dpHigh = drHigh.process(pAUCLiberal=minRate)
    dpLow = drLow.process(pAUCLiberal=minRate)

    #bootstrap 
    dpHigh.calculateConfidenceBootstrap(nBootstraps=10)

    #pair conditions 
    dpLow.calculateConfidenceBootstrap(10, pairKey="participantId", pairs=dpHigh.bootstrapPairs)

    #compare pAUC 
    dpHigh.comparePAUC(dpLow, useCovariance=True)

    assert dpHigh.pAUC == pytest.approx(0.03215431554156516, rel=1e-5)
    assert dpLow.pAUC == pytest.approx(0.03202288325642245, rel=1e-5)
    






    

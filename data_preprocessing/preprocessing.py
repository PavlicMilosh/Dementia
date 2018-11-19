import pandas as pd
import numpy as np


if __name__ == '__main__':
    adrc = pd.read_csv("../numerical_data/raw/ADRC Clinical Data.csv")
    adrc.replace(to_replace='.', value=np.nan, inplace=True)

    adrc.drop(
        labels=["Date", "Age", "acsparnt", "height", "weight", "primStudy", "acsStudy"],
        axis="columns", inplace=True)

    clinician_diagnosis = pd.read_csv("../numerical_data/raw/Clinician Diagnosis.csv")

    clinician_diagnosis.drop(
        labels=["Date", "Age", "DEMENTED", "MCIAMEM", "MCIAPLUS", "MCIAPLAN", "MCIAPATT", "MCIAPEX",
                "MCIAPVIS", "MCINON1", "MCIN1LAN", "MCIN1ATT", "MCIN1EX", "MCIN1VIS", "MCINON2", "MCIN2LAN",
                "MCIN2ATT", "MCIN2EX", "MCIN2VIS", "IMPNOMCI", "PROBAD", "PROBADIF", "POSSAD", "POSSADIF",
                "DLB", "DLBIF", "VASC", "VASCIF", "VASCPS", "VASCPSIF", "ALCDEM", "ALCDEMIF", "DEMUN",
                "DEMUNIF", "FTD", "FTDIF", "PPAPH", "PPAPHIF", "PNAPH", "SEMDEMAN", "SEMDEMAG", "PPAOTHR",
                "PSPIF", "CORTIF", "HUNTIF", "PRIONIF", "MEDSIF", "DYSILLIF", "DEPIF", "OTHPSYIF", "DOWNSIF",
                "PARKIF", "STROKIF", "HYCEPHIF", "BRNINJIF", "NEOPIF", "COGOTHX", "COGOTHIF", "COGOTH2X",
                "COGOTH2F", "COGOTH3X", "COGOTH3F"],
        axis="columns", inplace=True)

    # clinician_judgements = pd.read_csv("../numerical_data/raw/Clinician Judgements.csv")
    #
    # clinician_judgements.drop(labels=["Date", "Age", "DECAGE", "COGMEM", "COGJUDG", "COGLANG", "COGVIS", "COGATTN"],
    #                           axis="columns", inplace=True)
    # only 3 columns have enough values, all others are over 50% missing

    faqs = pd.read_csv("../numerical_data/raw/FAQs.csv")

    faqs.drop(
        labels=["Date", "Age"],
        axis="columns", inplace=True)

    freesurfers = pd.read_csv("../numerical_data/raw/FreeSurfers.csv")

    freesurfers.drop(
        labels=["FS Date", "Included T1s"],
        axis="columns", inplace=True)

    gds = pd.read_csv("../numerical_data/raw/GDS.csv")

    gds.drop(
        labels=["Date", "Age"],
        axis="columns", inplace=True)

    his_and_cvd = pd.read_csv("../numerical_data/raw/HIS and CVD.csv")

    his_and_cvd.drop(
        labels=["Date", "Age", "CVDIMAG1", "CVDIMAG2", "CVDIMAG3", "CVDIMAG4", "CVDIMAGX"],
        axis="columns", inplace=True)

    npi_q = pd.read_csv("../numerical_data/raw/NPI-Q.csv")

    npi_q.drop(
        labels=["Date", "Age", "INITIALS", "NPIQINFX", "DELSEV", "HALLSEV", "AGITSEV", "DEPDSEV", "ANXSEV", "ELATSEV",
                "APASEV", "DISNSEV", "IRRSEV", "MOTSEV", "NITESEV", "APPSEV"],
        axis="columns", inplace=True)

    # TODO: maybe sev is severity and is present only for patients who have the problem (fill with zeros or smth)?

    physical_neuro_findings = pd.read_csv("../numerical_data/raw/Physical Neuro Findings.csv")

    physical_neuro_findings.drop(
        labels=["Date", "Age"],
        axis="columns", inplace=True)

    sub_health_history = pd.read_csv("../numerical_data/raw/Subject Health History.csv")

    sub_health_history.drop(
        labels=["Date", "Age", "EXAMDATE", "CVOTHRX", "STROK1YR", "STROK2YR", "STROK3YR", "STROK4YR", "STROK5YR",
                "STROK6YR", "TIA1YR", "TIA2YR", "TIA3YR", "TIA4YR", "TIA5YR", "TIA6YR", "CBOTHRX", "PDYR", "PDOTHRYR",
                "NCOTHRX", "ABUSX", "PSYCDISX"],
        axis="columns", inplace=True)

    subjects = pd.read_csv("../numerical_data/raw/subjects.csv")

    subjects.drop(
        labels=["YOB"],
        axis="columns", inplace=True)

    # TODO: UPDRS has 50% missing values in every column (with 100% missing in every other column)
    # TODO: see if its worth using at all

    # number of rows in each csv
    # print(adrc.shape[0])                        # 6224 0
    # print(clinician_diagnosis.shape[0])         # 4089
    # print(faqs.shape[0])                        # 4089
    # print(freesurfers.shape[0])                 # 1984 -2
    # print(gds.shape[0])                         # 4089
    # print(his_and_cvd.shape[0])                 # 4089
    # print(npi_q.shape[0])                       # 4089
    # print(physical_neuro_findings.shape[0])     # 4089
    # print(sub_health_history.shape[0])          # 4089
    # print(subjects.shape[0])                    # 1098 -1

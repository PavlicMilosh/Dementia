import pandas as pd
import numpy as np



if __name__ == '__main__':
    adrc = pd.read_csv("../numerical_data/raw/ADRC Clinical Data.csv")

    '''
        Replacing '.' (missing) values with numpy nan
    '''
    adrc.replace(to_replace='.', value=np.nan, inplace=True)

    '''
        Dropping irrelevant labels from tables 
    '''
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

    '''
        Creating a list of ids for each table
    '''
    adrc_ids = adrc["ADRC_ADRCCLINICALDATA ID"].tolist()
    clinician_diagnosis_ids = clinician_diagnosis["UDS_D1DXDATA ID"].tolist()
    faqs_ids = faqs["UDS_B7FAQDATA ID"].tolist()
    gds_ids = gds["UDS_B6BEVGDSDATA ID"].tolist()
    his_and_cvd_ids = his_and_cvd["UDS_B2HACHDATA ID"].tolist()
    npi_q_ids = npi_q["UDS_B5BEHAVASDATA ID"].tolist()
    physical_neuro_findings_ids = physical_neuro_findings["UDS_B8EVALDATA ID"].tolist()
    sub_health_history_ids = sub_health_history["UDS_A5SUBHSTDATA ID"].tolist()

    all_ids = [clinician_diagnosis_ids, faqs_ids, gds_ids, his_and_cvd_ids, npi_q_ids, physical_neuro_findings_ids, sub_health_history_ids]
    new_ids = []
    for id_column in all_ids:
        new = list(map(lambda x: x.split("_")[0] + "_" + x.split("_")[2], id_column))
        new_ids += [new]

    adrc_ids = list(map(lambda x: x.split("_")[0] + "_" + x.split("_")[2], adrc_ids))

    '''
        Removing all instances from 4089 tables that don't contain labels in adrc table 
    '''

    # map where key is a column name which contains ids, and value is dataframe
    map = {
        "UDS_D1DXDATA ID": clinician_diagnosis,
        "UDS_B7FAQDATA ID": faqs,
        "UDS_B6BEVGDSDATA ID": gds,
        "UDS_B2HACHDATA ID": his_and_cvd,
        "UDS_B5BEHAVASDATA ID": npi_q,
        "UDS_B8EVALDATA ID": physical_neuro_findings,
        "UDS_A5SUBHSTDATA ID": sub_health_history
    }

    seti = set()
    for i in range(len(new_ids[1])):
        if new_ids[1][i] not in adrc_ids:
            seti.add(new_ids[1][i])

    for k, v in map.items():
        criterion = v[k].map(lambda i: i.split("_")[0] + "_" + i.split("_")[2] not in seti)
        map[k] = v[criterion]

    adrc = adrc[adrc["dx1"].notnull()]

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
    # all tables with 4089 have all the same instances (ids are all the same)

    # for i in new_ids:
    #     i.sort()

    # count = 0
    # for i in range(len(new_ids[0])):
    #     if new_ids[0][i] == new_ids[1][i] and new_ids[0][i] == new_ids[2][i] and new_ids[0][i] == new_ids[3][i] \
    #         and new_ids[0][i] == new_ids[4][i] and new_ids[0][i] == new_ids[5][i] and new_ids[0][i] == new_ids[6][i]:
    #         count += 1
    #         continue
    #     print(new_ids[0][i])
    #     print(new_ids[1][i])
    #     print(new_ids[2][i])
    #     print(new_ids[3][i])
    #     print(new_ids[4][i])
    #     print(new_ids[5][i])
    #     print(new_ids[6][i])
    #
    # print(count)

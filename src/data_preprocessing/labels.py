from enum import Enum

LABEL_MAP = {
    "Cognitively Normal": ["Cognitively normal", "No dementia"],
    "Uncertain Dementia": ["uncertain dementia", "Unc: ques. Impairment", "uncertain- possible NON AD dem",
                           "0.5 in memory only", "uncertain  possible NON AD dem", "Incipient demt PTP",
                           "Unc: impair reversible", "Incipient Non-AD dem"],
    "Alzheimer Dementia": ["AD dem w/depresss- not contribut", "AD Dementia", "AD dem distrubed social- with",
                           "AD dem w/CVD contribut", "AD dem visuospatial- prior", "AD dem Language dysf after",
                           "AD dem w/PDI after AD dem not contrib", "AD dem distrubed social- prior",
                           "AD dem distrubed social- after", "AD dem w/PDI after AD dem contribut",
                           "AD dem w/depresss  not contribut", "AD dem w/oth (list B) contribut",
                           "AD dem w/depresss- contribut", "AD dem w/oth (list B) not contrib",
                           "AD dem w/CVD not contrib",
                           "AD dem Language dysf prior", "AD dem Language dysf with",
                           "AD dem w/oth unusual features",
                           "AD dem cannot be primary", "AD dem w/oth unusual feat/subs demt",
                           "AD dem w/depresss  contribut", "AD dem visuospatial, after",
                           "AD dem visuospatial- with", "AD dem w/oth unusual features/demt on",
                           "AD dem w/Frontal lobe/demt at onset", "AD dem/FLD prior to AD dem",
                           "AD dem w/depresss, not contribut"],
    "Non AD Dementia":    ["Non AD dem- Other primary", "Vascular Demt- primary", "Frontotemporal demt. prim",
                           "DLBD- primary", "Dementia/PD- primary", "DAT", "Vascular Demt  primary", "DLBD- secondary",
                           "ProAph w/o dement", "Vascular Demt- secondary", "DLBD, primary",
                           "DAT w/depresss not contribut"]
}


class Label(Enum):

    COGNITIVELY_NORMAL = '0', 'Cognitively Normal'
    AD = '1', 'Alzheimer Dementia'
    UNCERTAIN = '2', 'Uncertain Dementia'
    NON_AD = '3', 'Non AD Dementia'
    DIAGNOSED = '4', 'Diagnosed Dementia'  # used on HEALTHY vs ALL


    def __new__(cls, value, name):
        member = object.__new__(cls)
        member._value_ = value
        member.fullname = name
        return member

    def __str__(self):
        return self.value

    def __int__(self):
        return int(self.value)

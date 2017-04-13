import re

alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

conv_table = {
    r'\\\'([A-Za-z])': ('ACEGIKLMNOPRSUWYZacegiklmnoprsuwyz', 'ÁĆÉǴÍḰĹḾŃÓṔŔŚÚẂÝŹáćéǵíḱĺḿńóṕŕśúẃýź'),  # acute accent
    r'\\`([A-Za-z])': ('AEINOUWYaeinouwy', 'ÀÈÌǸÒÙẀỲàèìǹòùẁỳ'),                                       # grave accent
    r'\\\^([A-Za-z])': ('ACEGHIJOSUWYZaceghijosuwyz', 'ÂĈÊĜĤÎĴÔŜÛŴŶẐâĉêĝĥîĵôŝûŵŷẑ'),                  # circumflex
    r'\\"([A-Za-z])': ('AEHIOUWXYaehiouwxy', 'ÄËḦÏÖÜẄẌŸäëḧïöüẅẍÿ'),                                   # diaeresis
    r'\{\\v ([A-Za-z])\}': ('ACDEGHIKNORSUZacdeghiknorsuz', 'ǍČĎĚǦȞǏǨŇǑŘŠǓŽǎčďěǧȟǐǩňǒřšǔž')           # caron
}

conv_table = {regex: dict(zip(*mapping)) for regex, mapping in conv_table.items()}


def conv_tex_diacritic(s):
    """
    \'A -> Á; \`A -> À; {\v A} -> Ǎ; \"A -> Ä; \^A -> Â
    Handles both upper and lower cases
    :param s: the string contains diacritics in tex format
    :return: the converted string with true diacritic characters
    """
    global conv_table
    for patt, mapping in conv_table.items():
        m = re.search(patt, s)
        if m:
            s = re.sub(patt, mapping[m.group(1)], s)
    return s


if __name__ == '__main__':
    s_list = ["K\\\"unneth formula",
              "Crit\\`ere de platitude par fibres: locally nilpotent case",
              "\\'Etale Morphisms of Schemes",
              "The {\\v C}ech complex"]
    for s in s_list:
        print('%s -> %s' % (s, conv_tex_diacritic(s)))

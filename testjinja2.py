#!python2
#coding=utf-8

import jinja2, codecs

params = []
params.append(
    dict(si=1, bi=2, tilt_s=3, azimuth_s=4, pathloss=5)
)
params.append(
    dict(si=1, bi=2, tilt_s=3, azimuth_s=4, pathloss=6)
)
tmpl = jinja2.Environment(
        loader=jinja2.FileSystemLoader('./')
        ).get_template("rsrpsinr_template.pyx")
source_code = tmpl.render(all_b_to_s_params=params)
with codecs.open("rsrpsinrfix.pyx", "w", "utf-8") as f:
    f.write(source_code)
from django import forms

from django.core.validators import MaxValueValidator, MinValueValidator

mar_choices =  [
    ('single', 'Single'),
    ('married', 'Married'),
    ('divorced', 'Divorced')
    ]
yes_no = [
    (1,'Yes'),
    (0,'No')
]
tel_choices = [
    (1, 'Telephone'),
    (0,'Cellular')
]

reg_choices =  [
    ('north', 'North'),
    ('south', 'South'),
    ('east','East'),
    ('west','West'),
    ('central','Central')
    ]
job_choices=[
    (7.0,'Services'),
    (0.0,'Admin'),
    (1.0,'Blue-collar'),
    (9.0,'Technician'),
    (3.0,'Housemaid'),
    (5.0,'Retired'),
    (4.0, 'Management'),
    (2.0,'Entrepreneur'),
    (10.0, 'Unemployed'),
    (8.0,'Student'),
    (6.0,'Self-employed')

]
week_choices = [
    (1, 'Monday'), 
    (2, "Tuesday"), 
    (3, "Wednesday"),
    (4, "Thursday"),
    (5,"Friday")
    ]

edu_choices=[
    (4,'High.school'),
    (3,'Basic.9y'),
    (5,'Professional.course'),
    (1,'Basic.4y'),
    (2,'Basic.6y'),
    (6,'University.degree'),
    (0,'Illiterate')
    ]

pout_choices=[
    ('success','Success'),
    ('failure','Failure'),
    ('nonexistent','Non Existent')
]

month_choices=[
    (1,'January'),
    (2,'February'),
    (3,'March'),
    (4,'April'),
    (5,'May'),
    (6,'June'),
    (7,'July'),
    (8,'August'),
    (9,'September'),
    (10,'October'),
    (11,'November'),
    (12,'December')
]

city_choices = [(0.0, 'Aberdeen'), (1.0, 'Abilene'), (2.0, 'Akron'), (3.0, 'Albuquerque'),
    (4.0, 'Alexandria'), (5.0, 'Allen'), (6.0, 'Allentown'), (7.0, 'Altoona'), (8.0, 'Amarillo'),
    (9.0, 'Anaheim'), (10.0, 'Andover'), (11.0, 'Ann Arbor'), (12.0, 'Antioch'), 
    (13.0, 'Apopka'), (14.0, 'Apple Valley'), (15.0, 'Appleton'), (16.0, 'Arlington'), 
    (17.0, 'Arlington Heights'), (18.0, 'Arvada'), (19.0, 'Asheville'), (20.0, 'Athens'),
    (21.0, 'Atlanta'), (22.0, 'Atlantic City'), (23.0, 'Auburn'), (24.0, 'Aurora'), (25.0, 'Austin'),
    (26.0, 'Avondale'), (27.0, 'Bakersfield'), (28.0, 'Baltimore'), (29.0, 'Bangor'), (30.0, 'Bartlett'),
    (31.0, 'Bayonne'), (32.0, 'Baytown'), (33.0, 'Beaumont'), (34.0, 'Bedford'), (35.0, 'Belleville'), 
    (36.0, 'Bellevue'), (37.0, 'Bellingham'), (38.0, 'Bethlehem'), (39.0, 'Beverly'), (40.0, 'Billings'), 
    (41.0, 'Bloomington'), (42.0, 'Boca Raton'), (43.0, 'Boise'), (44.0, 'Bolingbrook'), (45.0, 'Bossier City'),
    (46.0, 'Bowling Green'), (47.0, 'Boynton Beach'), (48.0, 'Bozeman'), (49.0, 'Brentwood'), (50.0, 'Bridgeton'), 
    (51.0, 'Bristol'), (52.0, 'Broken Arrow'), (53.0, 'Broomfield'), (54.0, 'Brownsville'), (55.0, 'Bryan'), 
    (56.0, 'Buffalo'), (57.0, 'Buffalo Grove'), (58.0, 'Bullhead City'), (59.0, 'Burbank'), (60.0, 'Burlington'),
    (61.0, 'Caldwell'), (62.0, 'Camarillo'), (63.0, 'Cambridge'), (64.0, 'Canton'), (65.0, 'Carlsbad'), 
    (66.0, 'Carol Stream'), (67.0, 'Carrollton'), (68.0, 'Cary'), (69.0, 'Cedar Hill'), (70.0, 'Cedar Rapids'),
    (71.0, 'Champaign'), (72.0, 'Chandler'), (73.0, 'Chapel Hill'), (74.0, 'Charlotte'), (75.0, 'Charlottesville'), 
    (76.0, 'Chattanooga'), (77.0, 'Chesapeake'), (78.0, 'Chester'), (79.0, 'Cheyenne'), (80.0, 'Chicago'), 
    (81.0, 'Chico'), (82.0, 'Chula Vista'), (83.0, 'Cincinnati'), (84.0, 'Citrus Heights'), (85.0, 'Clarksville'),
    (86.0, 'Cleveland'), (87.0, 'Clifton'), (88.0, 'Clinton'), (89.0, 'Clovis'), (90.0, 'Coachella'),
    (91.0, 'College Station'), (92.0, 'Colorado Springs'), (93.0, 'Columbia'), (94.0, 'Columbus'),
    (95.0, 'Commerce City'), (96.0, 'Concord'), (97.0, 'Conroe'), (98.0, 'Conway'), (99.0, 'Coon Rapids'), 
    (100.0, 'Coppell'), (101.0, 'Coral Gables'), (102.0, 'Coral Springs'), (103.0, 'Corpus Christi'), 
    (104.0, 'Costa Mesa'), (105.0, 'Cottage Grove'), (106.0, 'Covington'), (107.0, 'Cranston'), 
    (108.0, 'Cuyahoga Falls'), (109.0, 'Dallas'), (110.0, 'Danbury'), (111.0, 'Danville'), (112.0, 'Davis'), 
    (113.0, 'Daytona Beach'), (114.0, 'Dearborn'), (115.0, 'Dearborn Heights'), (116.0, 'Decatur'), 
    (117.0, 'Deer Park'), (118.0, 'Delray Beach'), (119.0, 'Deltona'), (120.0, 'Denver'), (121.0, 'Des Moines'), 
    (122.0, 'Des Plaines'), (123.0, 'Detroit'), (124.0, 'Dover'), (125.0, 'Draper'), (126.0, 'Dublin'), (127.0, 'Dubuque'), 
    (128.0, 'Durham'), (129.0, 'Eagan'), (130.0, 'East Orange'), (131.0, 'East Point'), (132.0, 'Eau Claire'), 
    (133.0, 'Edinburg'), (134.0, 'Edmond'), (135.0, 'Edmonds'), (136.0, 'El Cajon'), (137.0, 'El Paso'), 
    (138.0, 'Elkhart'), (139.0, 'Elmhurst'), (140.0, 'Elyria'), (141.0, 'Encinitas'), (142.0, 'Englewood'), 
    (143.0, 'Escondido'), (144.0, 'Eugene'), (145.0, 'Evanston'), (146.0, 'Everett'), (147.0, 'Fairfield'), 
    (148.0, 'Fargo'), (149.0, 'Farmington'), (150.0, 'Fayetteville'), (151.0, 'Florence'), (152.0, 'Fort Collins'), 
    (153.0, 'Fort Lauderdale'), (154.0, 'Fort Worth'), (155.0, 'Frankfort'), (156.0, 'Franklin'), (157.0, 'Freeport'), 
    (158.0, 'Fremont'), (159.0, 'Fresno'), (160.0, 'Frisco'), (161.0, 'Gaithersburg'), (162.0, 'Garden City'), 
    (163.0, 'Garland'), (164.0, 'Gastonia'), (165.0, 'Georgetown'), (166.0, 'Gilbert'), (167.0, 'Gladstone'), 
    (168.0, 'Glendale'), (169.0, 'Glenview'), (170.0, 'Goldsboro'), (171.0, 'Grand Island'), (172.0, 'Grand Prairie'), 
    (173.0, 'Grand Rapids'), (174.0, 'Grapevine'), (175.0, 'Great Falls'), (176.0, 'Greeley'), (177.0, 'Green Bay'), 
    (178.0, 'Greensboro'), (179.0, 'Greenville'), (180.0, 'Greenwood'), (181.0, 'Gresham'), (182.0, 'Grove City'), 
    (183.0, 'Gulfport'), (184.0, 'Hackensack'), (185.0, 'Hagerstown'), (186.0, 'Haltom City'), (187.0, 'Hamilton'), 
    (188.0, 'Hampton'), (189.0, 'Harlingen'), (190.0, 'Harrisonburg'), (191.0, 'Hattiesburg'), (192.0, 'Helena'), 
    (193.0, 'Hempstead'), (194.0, 'Henderson'), (195.0, 'Hendersonville'), (196.0, 'Hesperia'), (197.0, 'Hialeah'), 
    (198.0, 'Hickory'), (199.0, 'Highland Park'), (200.0, 'Hillsboro'), (201.0, 'Holland'), (202.0, 'Hollywood'), 
    (203.0, 'Holyoke'), (204.0, 'Homestead'), (205.0, 'Hoover'), (206.0, 'Hot Springs'), (207.0, 'Houston'), 
    (208.0, 'Huntington Beach'), (209.0, 'Huntsville'), (210.0, 'Independence'), (211.0, 'Indianapolis'), (212.0, 'Inglewood'), 
    (213.0, 'Iowa City'), (214.0, 'Irving'), (215.0, 'Jackson'), (216.0, 'Jacksonville'), (217.0, 'Jamestown'), 
    (218.0, 'Jefferson City'), (219.0, 'Johnson City'), (220.0, 'Jonesboro'), (221.0, 'Jupiter'), (222.0, 'Keller'), 
    (223.0, 'Kenner'), (224.0, 'Kenosha'), (225.0, 'Kent'), (226.0, 'Kirkwood'), (227.0, 'Kissimmee'), (228.0, 'Knoxville'), 
    (229.0, 'La Crosse'), (230.0, 'La Mesa'), (231.0, 'La Porte'), (232.0, 'La Quinta'), (233.0, 'Lafayette'), 
    (234.0, 'Laguna Niguel'), (235.0, 'Lake Charles'), (236.0, 'Lake Elsinore'), (237.0, 'Lake Forest'), 
    (238.0, 'Lakeland'), (239.0, 'Lakeville'), (240.0, 'Lakewood'), (241.0, 'Lancaster'), (242.0, 'Lansing'), 
    (243.0, 'Laredo'), (244.0, 'Las Cruces'), (245.0, 'Las Vegas'), (246.0, 'Laurel'), (247.0, 'Lawrence'), 
    (248.0, 'Lawton'), (249.0, 'Layton'), (250.0, 'League City'), (251.0, 'Lebanon'), (252.0, 'Lehi'), 
    (253.0, 'Leominster'), (254.0, 'Lewiston'), (255.0, 'Lincoln Park'), (256.0, 'Linden'), (257.0, 'Lindenhurst'), 
    (258.0, 'Little Rock'), (259.0, 'Littleton'), (260.0, 'Lodi'), (261.0, 'Logan'), (262.0, 'Long Beach'), (263.0, 'Longmont'), 
    (264.0, 'Longview'), (265.0, 'Lorain'), (266.0, 'Los Angeles'), (267.0, 'Louisville'), (268.0, 'Loveland'), 
    (269.0, 'Lowell'), (270.0, 'Lubbock'), (271.0, 'Macon'), (272.0, 'Madison'), (273.0, 'Malden'), (274.0, 'Manchester'), 
    (275.0, 'Manhattan'), (276.0, 'Mansfield'), (277.0, 'Manteca'), (278.0, 'Maple Grove'), (279.0, 'Margate'), 
    (280.0, 'Marietta'), (281.0, 'Marion'), (282.0, 'Marlborough'), (283.0, 'Marysville'), (284.0, 'Mason'), 
    (285.0, 'Mcallen'), (286.0, 'Medford'), (287.0, 'Medina'), (288.0, 'Melbourne'), (289.0, 'Memphis'), (290.0, 'Mentor'), 
    (291.0, 'Meriden'), (292.0, 'Meridian'), (293.0, 'Mesa'), (294.0, 'Mesquite'), (295.0, 'Miami'), (296.0, 'Middletown'), 
    (297.0, 'Midland'), (298.0, 'Milford'), (299.0, 'Milwaukee'), (300.0, 'Minneapolis'), (301.0, 'Miramar'), 
    (302.0, 'Mishawaka'), (303.0, 'Mission Viejo'), (304.0, 'Missoula'), (305.0, 'Missouri City'), (306.0, 'Mobile'), 
    (307.0, 'Modesto'), (308.0, 'Monroe'), (309.0, 'Montebello'), (310.0, 'Montgomery'), (311.0, 'Moorhead'), 
    (312.0, 'Moreno Valley'), (313.0, 'Morgan Hill'), (314.0, 'Morristown'), (315.0, 'Mount Pleasant'), (316.0, 'Mount Vernon'), 
    (317.0, 'Murfreesboro'), (318.0, 'Murray'), (319.0, 'Murrieta'), (320.0, 'Muskogee'), (321.0, 'Naperville'), 
    (322.0, 'Nashua'), (323.0, 'Nashville'), (324.0, 'New Albany'), (325.0, 'New Bedford'), (326.0, 'New Brunswick'), 
    (327.0, 'New Castle'), (328.0, 'New Rochelle'), (329.0, 'New York City'), (330.0, 'Newark'), (331.0, 'Newport News'), 
    (332.0, 'Niagara Falls'), (333.0, 'Noblesville'), (334.0, 'Norfolk'), (335.0, 'Normal'), (336.0, 'Norman'), 
    (337.0, 'North Charleston'), (338.0, 'North Las Vegas'), (339.0, 'North Miami'), (340.0, 'Norwich'), (341.0, 'Oak Park'), 
    (342.0, 'Oakland'), (343.0, 'Oceanside'), (344.0, 'Odessa'), (345.0, 'Oklahoma City'), (346.0, 'Olathe'), 
    (347.0, 'Olympia'), (348.0, 'Omaha'), (349.0, 'Ontario'), (350.0, 'Orange'), (351.0, 'Orem'), (352.0, 'Orland Park'), 
    (353.0, 'Orlando'), (354.0, 'Ormond Beach'), (355.0, 'Oswego'), (356.0, 'Overland Park'), (357.0, 'Owensboro'), 
    (358.0, 'Oxnard'), (359.0, 'Palatine'), (360.0, 'Palm Coast'), (361.0, 'Park Ridge'), (362.0, 'Parker'), (363.0, 'Parma'), 
    (364.0, 'Pasadena'), (365.0, 'Pasco'), (366.0, 'Passaic'), (367.0, 'Paterson'), (368.0, 'Pearland'), 
    (369.0, 'Pembroke Pines'), (370.0, 'Pensacola'), (371.0, 'Peoria'), (372.0, 'Perth Amboy'), (373.0, 'Pharr'), 
    (374.0, 'Philadelphia'), (375.0, 'Phoenix'), (376.0, 'Pico Rivera'), (377.0, 'Pine Bluff'), (378.0, 'Plainfield'), 
    (379.0, 'Plano'), (380.0, 'Plantation'), (381.0, 'Pleasant Grove'), (382.0, 'Pocatello'), (383.0, 'Pomona'), 
    (384.0, 'Pompano Beach'), (385.0, 'Port Arthur'), (386.0, 'Port Orange'), (387.0, 'Port Saint Lucie'), (388.0, 'Portage'), 
    (389.0, 'Portland'), (390.0, 'Providence'), (391.0, 'Provo'), (392.0, 'Pueblo'), (393.0, 'Quincy'), (394.0, 'Raleigh'), 
    (395.0, 'Rancho Cucamonga'), (396.0, 'Rapid City'), (397.0, 'Reading'), (398.0, 'Redding'), (399.0, 'Redlands'), 
    (400.0, 'Redmond'), (401.0, 'Redondo Beach'), (402.0, 'Redwood City'), (403.0, 'Reno'), (404.0, 'Renton'), 
    (405.0, 'Revere'), (406.0, 'Richardson'), (407.0, 'Richmond'), (408.0, 'Rio Rancho'), (409.0, 'Riverside'), 
    (410.0, 'Rochester'), (411.0, 'Rochester Hills'), (412.0, 'Rock Hill'), (413.0, 'Rockford'), (414.0, 'Rockville'), 
    (415.0, 'Rogers'), (416.0, 'Rome'), (417.0, 'Romeoville'), (418.0, 'Roseville'), (419.0, 'Roswell'), 
    (420.0, 'Round Rock'), (421.0, 'Royal Oak'), (422.0, 'Sacramento'), (423.0, 'Saginaw'), (424.0, 'Saint Charles'), 
    (425.0, 'Saint Cloud'), (426.0, 'Saint Louis'), (427.0, 'Saint Paul'), (428.0, 'Saint Peters'), (429.0, 'Saint Petersburg'), 
    (430.0, 'Salem'), (431.0, 'Salinas'), (432.0, 'Salt Lake City'), (433.0, 'San Angelo'), (434.0, 'San Antonio'), 
    (435.0, 'San Bernardino'), (436.0, 'San Clemente'), (437.0, 'San Diego'), (438.0, 'San Francisco'), (439.0, 'San Gabriel'), 
    (440.0, 'San Jose'), (441.0, 'San Luis Obispo'), (442.0, 'San Marcos'), (443.0, 'San Mateo'), (444.0, 'Sandy Springs'), 
    (445.0, 'Sanford'), (446.0, 'Santa Ana'), (447.0, 'Santa Barbara'), (448.0, 'Santa Clara'), (449.0, 'Santa Fe'), 
    (450.0, 'Santa Maria'), (451.0, 'Scottsdale'), (452.0, 'Seattle'), (453.0, 'Sheboygan'), (454.0, 'Shelton'), 
    (455.0, 'Sierra Vista'), (456.0, 'Sioux Falls'), (457.0, 'Skokie'), (458.0, 'Smyrna'), (459.0, 'South Bend'), 
    (460.0, 'Southaven'), (461.0, 'Sparks'), (462.0, 'Spokane'), (463.0, 'Springdale'), (464.0, 'Springfield'), 
    (465.0, 'Sterling Heights'), (466.0, 'Stockton'), (467.0, 'Suffolk'), (468.0, 'Summerville'), (469.0, 'Sunnyvale'), 
    (470.0, 'Superior'), (471.0, 'Tallahassee'), (472.0, 'Tamarac'), (473.0, 'Tampa'), (474.0, 'Taylor'), (475.0, 'Temecula'), 
    (476.0, 'Tempe'), (477.0, 'Texarkana'), (478.0, 'Texas City'), (479.0, 'The Colony'), (480.0, 'Thomasville'), 
    (481.0, 'Thornton'), (482.0, 'Thousand Oaks'), (483.0, 'Tigard'), (484.0, 'Tinley Park'), (485.0, 'Toledo'), 
    (486.0, 'Torrance'), (487.0, 'Trenton'), (488.0, 'Troy'), (489.0, 'Tucson'), (490.0, 'Tulsa'), (491.0, 'Tuscaloosa'), 
    (492.0, 'Twin Falls'), (493.0, 'Tyler'), (494.0, 'Urbandale'), (495.0, 'Utica'), (496.0, 'Vacaville'), (497.0, 'Vallejo'), 
    (498.0, 'Vancouver'), (499.0, 'Vineland'), (500.0, 'Virginia Beach'), (501.0, 'Visalia'), (502.0, 'Waco'), 
    (503.0, 'Warner Robins'), (504.0, 'Warwick'), (505.0, 'Washington'), (506.0, 'Waterbury'), (507.0, 'Waterloo'), 
    (508.0, 'Watertown'), (509.0, 'Waukesha'), (510.0, 'Wausau'), (511.0, 'Waynesboro'), (512.0, 'West Allis'), 
    (513.0, 'West Jordan'), (514.0, 'West Palm Beach'), (515.0, 'Westfield'), (516.0, 'Westland'), (517.0, 'Westminster'), 
    (518.0, 'Wheeling'), (519.0, 'Whittier'), (520.0, 'Wichita'), (521.0, 'Wilmington'), (522.0, 'Wilson'), (523.0, 'Woodbury'), 
    (524.0, 'Woodland'), (525.0, 'Woodstock'), (526.0, 'Woonsocket'), (527.0, 'Yonkers'), (528.0, 'York'), (529.0, 'Yucaipa'), 
    (530.0, 'Yuma')]

state_choices=[(0.0, 'Alabama'), (1.0, 'Arizona'), (2.0, 'Arkansas'), (3.0, 'California'), (4.0, 'Colorado'), 
    (5.0, 'Connecticut'), (6.0, 'Delaware'), (7.0, 'District of Columbia'), (8.0, 'Florida'), (9.0, 'Georgia'), 
    (10.0, 'Idaho'), (11.0, 'Illinois'), (12.0, 'Indiana'), (13.0, 'Iowa'), (14.0, 'Kansas'), (15.0, 'Kentucky'), 
    (16.0, 'Louisiana'), (17.0, 'Maine'), (18.0, 'Maryland'), (19.0, 'Massachusetts'), (20.0, 'Michigan'), (21.0, 'Minnesota'), 
    (22.0, 'Mississippi'), (23.0, 'Missouri'), (24.0, 'Montana'), (25.0, 'Nebraska'), (26.0, 'Nevada'), (27.0, 'New Hampshire'), 
    (28.0, 'New Jersey'), (29.0, 'New Mexico'), (30.0, 'New York'), (31.0, 'North Carolina'), (32.0, 'North Dakota'), 
    (33.0, 'Ohio'), (34.0, 'Oklahoma'), (35.0, 'Oregon'), (36.0, 'Pennsylvania'), (37.0, 'Rhode Island'), 
    (38.0, 'South Carolina'), (39.0, 'South Dakota'), (40.0, 'Tennessee'), (41.0, 'Texas'), (42.0, 'Utah'), (43.0, 'Virginia'), 
    (44.0, 'Washington'), (45.0, 'Wisconsin'), (46.0, 'Wyoming')]

class Form1(forms.Form):
    age  = forms.FloatField(label = 'Age', 
                               required = True,
                               validators = [
                                MaxValueValidator(100),
                                MinValueValidator(18)
                               ] )
    marital_status = forms.CharField(label='Marital Status ', widget=forms.Select(choices=mar_choices))
    has_credit = forms.FloatField(label='Has credit in default? ', widget=forms.Select(choices=yes_no))
    housing_loan  = forms.FloatField(label='Has Housing Loan? ', widget=forms.Select(choices=yes_no))
    personal_loan = forms.FloatField(label='Has Personal Loan?', widget=forms.Select(choices=yes_no))
    contact_mode = forms.FloatField(label='Conatct mode ', widget=forms.Select(choices=tel_choices))
    duration  = forms.FloatField(label = 'Duration of last contact in sec',
                               required = True,
                               validators = [
                                MinValueValidator(0)
                               ] )
    campaign  = forms.FloatField(label = 'No of contacts during Campaign', 
                               required = True,
                               validators = [
                                MinValueValidator(0)
                               ] )
    pdays  = forms.FloatField(label = 'No. of days passed Last contacted', 
                               required = True,
                               validators = [
                                MaxValueValidator(999),
                                MinValueValidator(0)
                               ] )
    previous  = forms.FloatField(label = 'No. of times previously contacted', 
                               required = True,
                               validators = [
                                MinValueValidator(0)
                               ] )
    region = forms.CharField(label='Region ', widget=forms.Select(choices=reg_choices))

    job= forms.FloatField(label='Job', widget=forms.Select(choices=job_choices))

    month= forms.FloatField(label='Month', widget=forms.Select(choices=month_choices))

    day_of_week= forms.FloatField(label='Day of week', widget=forms.Select(choices=week_choices))
 
    education = forms.FloatField(label='Education', widget=forms.Select(choices=edu_choices))

    city_name = forms.FloatField(label='City Name', widget=forms.Select(choices=city_choices))

    state_name = forms.FloatField(label='State Name', widget=forms.Select(choices=state_choices))

    poutcome = forms.CharField(label='Outcome of previous Campaign', widget=forms.Select(choices=pout_choices))

    emp_var  = forms.FloatField(label = 'Employee Variation Rate')

    cons_price_idx  = forms.FloatField(label = 'Consumer Price Index')
    
    cons_conf_idx  = forms.FloatField(label = 'Consumer Confidence Index')
                    
    euribor3m  = forms.FloatField(label = 'Euribor 3 month Rate')

    nr_employed = forms.FloatField(label = 'No of employees')
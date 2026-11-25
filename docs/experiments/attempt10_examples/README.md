# Example panels: attempt10, fold 0

Made by `project_a/cli/examples.py --share` (839 validation images scored; the full-size PNGs and scores.csv stay with the run on SCC). Quality = 0.5 IoU + 0.3 clDice + 0.2 Boundary IoU, per image, in percent.

| Mean of per-image quality | K=0 (H0) | K=1 | K=2 | K=4 | K=4, z reset |
|---|---|---|---|---|---|
| All scored images | 48.2 | 58.4 | 58.8 | 59.0 | 54.4 |

This averages each image's own score, so near-empty images count as much as busy ones. The fold metric in the results docs counts pixels over all images together and is different.

How to read a panel (three rows of four tiles):

- Row 1: image, truth (LoveDA classes, road = yellow), prediction at K=0 and K=1.
- Row 2: prediction at K=2, K=4, K=4 with z reset before every step, and how much the road probability changes when z is reset (black = no change, orange = large).
- Row 3: road probability at K=0, 1, 2, 4 (dark purple = 0, yellow = 1).
- Prediction colors: white = road hit, red = false road, blue = missed road, black = background, gray = no-data.

Categories: [worst](#worst), [best](#best), [improves](#improves), [no_gain](#no_gain), [late](#late), [uses_z](#uses_z)

## Where it helps and where it hurts

Change from H0 (K=0) to K=4, as means of per-image scores. Images with under 1% road are left out of every row but the last; there an empty truth and an empty prediction score 100, so that row mostly measures removed false road. The last column is the share of images that get worse by more than 1 point.

| Group | Images | Quality H0 -> K=4 | IoU | clDice | BIoU | Worse by >1 |
|---|---|---|---|---|---|---|
| Rural | 217 | 50.3 -> 59.6 (**+9.3**) | +9.4 | +8.9 | +9.7 | 9% |
| Urban | 310 | 53.9 -> 57.3 (**+3.4**) | +2.4 | +7.3 | +0.1 | 17% |
| Rural, thin roads (1-3% road) | 126 | 46.2 -> 57.7 (**+11.5**) | +12.0 | +9.3 | +13.5 | 10% |
| Rural, medium roads (3-8% road) | 57 | 52.9 -> 60.5 (**+7.6**) | +7.4 | +8.0 | +7.5 | 9% |
| Rural, dense roads (8-100% road) | 34 | 60.9 -> 65.2 (**+4.3**) | +3.4 | +9.3 | -0.9 | 9% |
| Urban, thin roads (1-3% road) | 46 | 39.4 -> 44.5 (**+5.1**) | +3.3 | +9.0 | +4.0 | 20% |
| Urban, medium roads (3-8% road) | 93 | 48.7 -> 53.7 (**+5.1**) | +3.9 | +9.2 | +1.8 | 18% |
| Urban, dense roads (8-100% road) | 171 | 60.7 -> 62.8 (**+2.0**) | +1.4 | +5.8 | -1.9 | 15% |
| H0 quality 0-20 | 28 | 6.7 -> 20.2 (**+13.5**) | +12.6 | +21.4 | +3.9 | 14% |
| H0 quality 20-40 | 82 | 32.2 -> 43.5 (**+11.2**) | +10.8 | +14.8 | +6.9 | 10% |
| H0 quality 40-60 | 213 | 52.0 -> 57.6 (**+5.6**) | +4.6 | +7.5 | +5.5 | 13% |
| H0 quality 60-100 | 204 | 67.3 -> 70.1 (**+2.8**) | +2.9 | +3.8 | +1.3 | 16% |
| Almost no road (< 1%) | 312 | 41.0 -> 60.3 (**+19.3**) | +19.2 | +19.4 | +19.4 | 7% |

## worst

Lowest quality at K=4.

| # | Image | K=0 | K=1 | K=2 | K=4 | K=4 z reset | K=4 IoU | K=4 clDice | K=4 BIoU | z flip % | Road % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | [rural/2614](worst/01_rural_2614.jpg) | 0.9 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| 2 | [rural/2639](worst/02_rural_2639.jpg) | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.6 |
| 3 | [rural/2708](worst/03_rural_2708.jpg) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.2 | 1.4 |
| 4 | [rural/2974](worst/04_rural_2974.jpg) | 1.0 | 0.0 | 0.0 | 0.0 | 5.6 | 0.0 | 0.0 | 0.0 | 0.5 | 2.2 |
| 5 | [urban/1610](worst/05_urban_1610.jpg) | 0.4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 13.1 |
| 6 | [urban/2207](worst/06_urban_2207.jpg) | 0.2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.2 | 2.0 |

**1. rural/2614**

![rural/2614](worst/01_rural_2614.jpg)

**2. rural/2639**

![rural/2639](worst/02_rural_2639.jpg)

**3. rural/2708**

![rural/2708](worst/03_rural_2708.jpg)

**4. rural/2974**

![rural/2974](worst/04_rural_2974.jpg)

**5. urban/1610**

![urban/1610](worst/05_urban_1610.jpg)

**6. urban/2207**

![urban/2207](worst/06_urban_2207.jpg)

## best

Highest quality at K=4.

| # | Image | K=0 | K=1 | K=2 | K=4 | K=4 z reset | K=4 IoU | K=4 clDice | K=4 BIoU | z flip % | Road % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | [rural/1204](best/01_rural_1204.jpg) | 62.7 | 82.9 | 83.4 | 83.9 | 81.5 | 84.2 | 97.6 | 62.3 | 0.1 | 1.1 |
| 2 | [rural/3445](best/02_rural_3445.jpg) | 61.5 | 81.5 | 83.5 | 83.6 | 77.8 | 85.0 | 99.8 | 55.6 | 0.1 | 1.1 |
| 3 | [rural/734](best/03_rural_734.jpg) | 65.0 | 83.4 | 83.4 | 82.9 | 82.0 | 84.2 | 97.3 | 58.2 | 0.1 | 1.0 |
| 4 | [rural/697](best/04_rural_697.jpg) | 68.3 | 83.0 | 82.5 | 82.1 | 84.2 | 82.8 | 99.1 | 54.9 | 0.1 | 1.4 |
| 5 | [rural/3044](best/05_rural_3044.jpg) | 44.7 | 78.3 | 80.2 | 82.0 | 79.2 | 83.9 | 99.9 | 50.2 | 0.1 | 1.1 |
| 6 | [rural/259](best/06_rural_259.jpg) | 59.6 | 80.9 | 80.9 | 80.8 | 79.4 | 87.3 | 99.5 | 36.8 | 0.3 | 2.7 |

**1. rural/1204**

![rural/1204](best/01_rural_1204.jpg)

**2. rural/3445**

![rural/3445](best/02_rural_3445.jpg)

**3. rural/734**

![rural/734](best/03_rural_734.jpg)

**4. rural/697**

![rural/697](best/04_rural_697.jpg)

**5. rural/3044**

![rural/3044](best/05_rural_3044.jpg)

**6. rural/259**

![rural/259](best/06_rural_259.jpg)

## improves

Largest gain from K=0 (H0) to K=4.

| # | Image | K=0 | K=1 | K=2 | K=4 | K=4 z reset | K=4 IoU | K=4 clDice | K=4 BIoU | z flip % | Road % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | [urban/2378](improves/01_urban_2378.jpg) | 1.5 | 45.8 | 55.4 | 59.7 | 52.3 | 63.2 | 89.3 | 6.7 | 1.1 | 4.1 |
| 2 | [urban/4019](improves/02_urban_4019.jpg) | 32.9 | 69.3 | 70.1 | 70.4 | 63.7 | 68.1 | 93.9 | 41.0 | 0.8 | 2.8 |
| 3 | [rural/3044](improves/03_rural_3044.jpg) | 44.7 | 78.3 | 80.2 | 82.0 | 79.2 | 83.9 | 99.9 | 50.2 | 0.1 | 1.1 |
| 4 | [rural/3023](improves/04_rural_3023.jpg) | 9.3 | 43.8 | 44.5 | 44.3 | 40.2 | 41.9 | 66.8 | 16.6 | 0.5 | 1.0 |
| 5 | [urban/1669](improves/05_urban_1669.jpg) | 3.1 | 32.3 | 37.3 | 38.0 | 42.6 | 37.5 | 61.6 | 4.0 | 3.9 | 7.7 |
| 6 | [rural/1322](improves/06_rural_1322.jpg) | 38.0 | 66.2 | 70.6 | 72.4 | 67.0 | 75.0 | 95.0 | 31.7 | 0.3 | 1.1 |

**1. urban/2378**

![urban/2378](improves/01_urban_2378.jpg)

**2. urban/4019**

![urban/4019](improves/02_urban_4019.jpg)

**3. rural/3044**

![rural/3044](improves/03_rural_3044.jpg)

**4. rural/3023**

![rural/3023](improves/04_rural_3023.jpg)

**5. urban/1669**

![urban/1669](improves/05_urban_1669.jpg)

**6. rural/1322**

![rural/1322](improves/06_rural_1322.jpg)

## no_gain

Smallest gain from K=0 to K=4 (the refiner makes these worse).

| # | Image | K=0 | K=1 | K=2 | K=4 | K=4 z reset | K=4 IoU | K=4 clDice | K=4 BIoU | z flip % | Road % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | [rural/3306](no_gain/01_rural_3306.jpg) | 46.6 | 23.0 | 18.8 | 18.1 | 52.3 | 16.6 | 30.4 | 3.4 | 1.0 | 2.5 |
| 2 | [urban/1449](no_gain/02_urban_1449.jpg) | 51.9 | 0.9 | 12.2 | 25.0 | 0.9 | 20.4 | 48.4 | 1.2 | 0.5 | 1.9 |
| 3 | [urban/1933](no_gain/03_urban_1933.jpg) | 56.0 | 34.1 | 34.0 | 34.1 | 41.1 | 31.6 | 59.5 | 2.1 | 1.6 | 14.7 |
| 4 | [urban/3868](no_gain/04_urban_3868.jpg) | 70.3 | 59.6 | 56.9 | 57.2 | 39.7 | 61.4 | 85.2 | 4.8 | 12.0 | 10.8 |
| 5 | [urban/3660](no_gain/05_urban_3660.jpg) | 50.1 | 39.5 | 37.9 | 37.5 | 34.6 | 35.9 | 60.9 | 6.5 | 1.6 | 3.0 |
| 6 | [rural/2996](no_gain/06_rural_2996.jpg) | 58.7 | 45.1 | 46.7 | 46.4 | 44.1 | 42.4 | 71.8 | 18.1 | 0.4 | 1.5 |

**1. rural/3306**

![rural/3306](no_gain/01_rural_3306.jpg)

**2. urban/1449**

![urban/1449](no_gain/02_urban_1449.jpg)

**3. urban/1933**

![urban/1933](no_gain/03_urban_1933.jpg)

**4. urban/3868**

![urban/3868](no_gain/04_urban_3868.jpg)

**5. urban/3660**

![urban/3660](no_gain/05_urban_3660.jpg)

**6. rural/2996**

![rural/2996](no_gain/06_rural_2996.jpg)

## late

Largest gain from K=1 to K=4, i.e. where steps 2-4 matter most.

| # | Image | K=0 | K=1 | K=2 | K=4 | K=4 z reset | K=4 IoU | K=4 clDice | K=4 BIoU | z flip % | Road % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | [urban/1449](late/01_urban_1449.jpg) | 51.9 | 0.9 | 12.2 | 25.0 | 0.9 | 20.4 | 48.4 | 1.2 | 0.5 | 1.9 |
| 2 | [urban/2378](late/02_urban_2378.jpg) | 1.5 | 45.8 | 55.4 | 59.7 | 52.3 | 63.2 | 89.3 | 6.7 | 1.1 | 4.1 |
| 3 | [urban/2039](late/03_urban_2039.jpg) | 13.0 | 29.2 | 39.1 | 37.4 | 40.1 | 33.8 | 67.6 | 1.1 | 0.7 | 3.7 |
| 4 | [urban/1464](late/04_urban_1464.jpg) | 20.7 | 9.5 | 16.2 | 17.7 | 6.7 | 15.4 | 30.1 | 4.9 | 1.7 | 2.2 |
| 5 | [urban/3934](late/05_urban_3934.jpg) | 19.4 | 31.2 | 36.9 | 37.6 | 36.9 | 28.9 | 63.7 | 20.5 | 1.2 | 2.7 |
| 6 | [rural/1322](late/06_rural_1322.jpg) | 38.0 | 66.2 | 70.6 | 72.4 | 67.0 | 75.0 | 95.0 | 31.7 | 0.3 | 1.1 |

**1. urban/1449**

![urban/1449](late/01_urban_1449.jpg)

**2. urban/2378**

![urban/2378](late/02_urban_2378.jpg)

**3. urban/2039**

![urban/2039](late/03_urban_2039.jpg)

**4. urban/1464**

![urban/1464](late/04_urban_1464.jpg)

**5. urban/3934**

![urban/3934](late/05_urban_3934.jpg)

**6. rural/1322**

![rural/1322](late/06_rural_1322.jpg)

## uses_z

Most K=4 pixels flip when the memory z is reset before every step.

| # | Image | K=0 | K=1 | K=2 | K=4 | K=4 z reset | K=4 IoU | K=4 clDice | K=4 BIoU | z flip % | Road % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | [urban/3868](uses_z/01_urban_3868.jpg) | 70.3 | 59.6 | 56.9 | 57.2 | 39.7 | 61.4 | 85.2 | 4.8 | 12.0 | 10.8 |
| 2 | [urban/2293](uses_z/02_urban_2293.jpg) | 40.9 | 38.1 | 39.3 | 40.4 | 41.3 | 44.4 | 59.9 | 0.9 | 5.7 | 6.1 |
| 3 | [rural/2988](uses_z/03_rural_2988.jpg) | 53.1 | 50.7 | 47.9 | 49.3 | 65.2 | 54.0 | 73.5 | 1.2 | 5.5 | 22.5 |
| 4 | [urban/1552](uses_z/04_urban_1552.jpg) | 62.0 | 66.6 | 65.7 | 65.4 | 68.5 | 76.1 | 88.9 | 3.3 | 4.8 | 29.3 |
| 5 | [urban/2486](uses_z/05_urban_2486.jpg) | 51.2 | 56.8 | 57.4 | 57.3 | 58.8 | 65.2 | 81.6 | 1.2 | 4.7 | 25.0 |
| 6 | [urban/1757](uses_z/06_urban_1757.jpg) | 45.5 | 43.9 | 45.6 | 45.8 | 38.8 | 44.8 | 73.2 | 7.1 | 4.4 | 6.5 |

**1. urban/3868**

![urban/3868](uses_z/01_urban_3868.jpg)

**2. urban/2293**

![urban/2293](uses_z/02_urban_2293.jpg)

**3. rural/2988**

![rural/2988](uses_z/03_rural_2988.jpg)

**4. urban/1552**

![urban/1552](uses_z/04_urban_1552.jpg)

**5. urban/2486**

![urban/2486](uses_z/05_urban_2486.jpg)

**6. urban/1757**

![urban/1757](uses_z/06_urban_1757.jpg)

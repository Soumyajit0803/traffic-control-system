# Traffic Control System

## Introduction
Urban traffic management and road safety remain critical challenges in developing smart cities. Conventional traffic light systems rely on fixed timing cycles, often failing to adapt to real-time variations in vehicle and pedestrian densities. Simultaneously, poor road conditions such as potholes significantly increase accident risks and reduce driving comfort. In this project, we propose an upgraded version of YOLOv8-based Intelligent Traffic Control System which is enhanced with ADAS (Advanced Driver Assistance System) data for real-time road surface monitoring. 

The system dynamically adjusts traffic signal durations based on detected traffic and pedestrian densities while simultaneously determining safe speed limits using acceleration and gyroscope data derived from ADAS sensors. This hybrid approach combines camera feed in pictures and sensor analytics to achieve adaptive, safety-oriented traffic control, contributing to more resilient and context-aware urban mobility.

## Past Literature
1. **Smartphone / IMU-based pothole detection (threshold + ML).**  
Many works use accelerometer + gyroscope from phones to detect potholes via RMS/peak detection, handcrafted features, or lightweight ML (SVM, RF). These are real-time, low-cost and robust when using sliding windows and robust scaling.
2. **YOLO (real-time CV) for traffic monitoring.**  
YOLO (and recent YOLOv8) is widely used for real-time vehicle/pedestrian detection and counting; many works build traffic-flow estimators, vehicle classification, and trackers (SORT/ByteTrack) on top of YOLO for live signal control.
3. **BATCS**
The Bengaluru traffic police has announced deployment of the Bengaluru Adaptive Traffic Control System(BATCS) designed to optimise traffic flow and reduce delays accross cities. [(More info)](https://timesofindia.indiatimes.com/auto/policy-and-industry/bengaluru-police-deploys-ai-based-traffic-control-system-heres-how-it-works/articleshow/114139000.cms)
4. **Roughness to speed relationship (empirical evidence).**  
Econometric work shows road roughness (IRI) correlates with lower vehicular speeds; one study finds **≈11% decrease in average speed per +1 SD increase in IRI**, a useful empirical anchor for mapping roughness → speed reduction. [(Link to paper)](https://www.nber.org/system/files/working_papers/w29176/w29176.pdf)

## Implementation
This project integrates **YOLOv8-based traffic monitoring** with **ADAS (Advanced Driver Assistance System) sensor analytics** to create an intelligent traffic management and safety system. The model dynamically adjusts **traffic light durations** and **speed limits** based on real-time conditions which include vehicle density, pedestrian presence, and road surface irregularities(potholes). By combining vision-based and sensor-based intelligence, it aims to enhance both **traffic efficiency** and **road safety**.
### Counting task: YOLOv8m model [(code)](./YOLOv8_density_tracker.ipynb)
The current implementation uses a custom wrapper around YOLOv8m model from Ultralytics to count the number of pedestrians and vehicles from camera image feed obtained every 15 minutes. 
The custom wrapper breaks the image into a 2x2 matrix and applies counting on all 4 of cropped parts separately, to enhance the counting accuracy. 
Currently, the untrained YOLOv8m model with custom parameters operated under the setup mentioned above gives an accuracy of around 80% which translates to missing out on miscalculation of only 5-6 pedestrians/vehicles.

**Proposed Signal timing calculation:**
1. **Defining constants**

	$S_{min} = 30s$

	$S_{max} = 120s$

	$\alpha = 0.8, \beta = 0.5, \gamma = 0.6$

	$\lambda = 0.8$ (smoothing factor)

	$T_{t-1}$ = Previous signal time

	$D_t$ = Traffic density

	$P_t$ = People density$

	$I_t$ = Road Irregularity Score

	Here, $\alpha$, $\beta$ and $\gamma$ denote the weights for people, traffic and irregularity scores that together help determine the signal timing

2. **Timing Computation**
	
	Following is the calculation of green signal timing. For red signal, we just swap the $D_t$ and $P_t$ values, keeping the formula same.

	$R_x = clip(\alpha*D_t - \beta*P_t + \gamma*I_t, 0, 1)$

	Where

	$
	{clip}(x, 0, 1) =
	\begin{cases}
	0, & x < 0 \\[6pt]
	x, & 0 \le x \le 1 \\[6pt]
	1, & x > 1
	\end{cases}
	$

	To keep the value within $S_{min}$ and $S_{max}$,

	$R_t = S_{min} + (S_{max} - S_{min}) * R_x$

	Finally, to avoid abrupt fluctuations we apply smoothing. Without it, even small sensor changes in traffic, pedestrians, or road irregularity could cause large, unstable shifts.

	**$T_t = \lambda*T_{t-1} + (1-\lambda)*R_t$**







### Road irregularity tracker: ADAS [(code)](./road_irregularity_tracker.ipynb)

For this we need a basic ADAS sensor that can provide 3-axis data of accelerometer and gyroscope separately for detecting road irregularity.

**The proposed road irregularity index calculation:**
1. **choosing parameters**

	$acc_z = accelerometerZ$

	$gyro_z = \sqrt{gyroX^2 + gyroY^2 + gyroZ^2}$

	> `gyroX` and `gyroY` would indicate pitching or rolling (e.g., bumps or banking turns). `gyroZ` tracks how sharply the vehicle turns (left/right).

	> Road roughness primarily produces vertical vibrations. Hence only `accelerometerZ` has been considered.

2. **Taking sliding window RMS of acceleration and gyro**
	$acc\_z\_rms= sliding\_rms(acc_z)$

	$gyro\_z\_rms= sliding\_rms(gyro_z)$

	RMS measures the energy of the signal. Taking it over a sliding window helps analyse how it changes over time.
  
	Window here is a fixed length of consecutive samples. 
	> If the sampling rate is e.g. 50 Hz, there will be 50 samples per window.

3. **Roughness Index (R\)**
	Combine the RMS signals of acceleration and gyro to produce roughness index. Weighted combination takes place, with more weight given to acceleration changes because on encountering a pothole the acceleration is more affected.

	$W_A = 0.7$
	
	$W_G = 0.3$

	$R_{raw} = W_A*acc\_z\_rms + W_G*gyro\_z\_rms$

4. **z-scoring Roughness index ($R_z$)**
	> Raw roughness index depends on sensor scaling and vehicle type. A sensor mounted loosely may detect bigger vibrations. A smooth car suspension reduces the same bump's reading. Hence we normalise the value for uniformity.
	
	$R_z = \frac{R_{\text{raw}} - \mu_R}{\sigma_R}$
	
	$R_z$ is out final, time-varying, **normalized signal** showing how roughness changes over time — but now expressed in `standard deviation units`. Interpretation:
	
	- $R_z(t) = 0$ : average road vibration
	- $R_z(t) > 0$ : rougher than average
	- $R_z(t) < 0$ : smoother than average

5. **Roughness score over the segment and calculation of speed limit**
	Once we have the signal representing the roughness over the segment, we need one representative number for scoring the segment of road based on roughness. The score directly determines how much the speed limit should be decreased from its maximum value.
	For the score, we take the $85^{th}$ percentile value of the signal $R_z(t)$. Using the **85th percentile** keeps the focus on the **rougher end** of the distribution. Taking an average will, instead, smoothen out dangerous spikes.
	
	$R_{score} = np.percentile(R_z,  85)$

	$S_{regulated} = S_{base} * max(f_{min}, 1-k*R_{score})$
	Where 
	* $S_{base}$ = 70 kmph, can be tweaked based on road type
	* $f_{min}$ = 0.3, the lower bound. Speed cannot decrease below 30% of the base.
	* $k$ = 0.11, speed reduction per roughness unit. Based on [this data](https://www.nber.org/system/files/working_papers/w29176/w29176.pdf).

6. **Roughness score as a determinant in traffic control**

	Since the roughness of the road affects vehicle speed, the traffic control needs to be altered for **longer green phases** to compensate for lower feasible driving speeds and increased vehicle delay. Additionally, the road can be restricted for trucks/buses/any other heavy vehicles that may aggravate the already damaged road condition.

The system assumes a decent cloud infrastructure that will collect the ADAS data from vehicles and feed the final computed data back to the display boards near traffic lights.

## Workflow
![Workflow](traffic-control.svg)



# Future Scope
1. **Tuning the parameters:** Currently the parameters used are purely based on logic and there is no proof behind it. The control system can be refined with feedback from drivers which cab be probably used to train a model.
2. **Upgradation to multiple lanes:** The traffic control can be upgraded to manage traffic for multiple lanes crossing each other. New determinants include the calculation of traffic and pedestrian densities alongwith road condition for other crossing lanes.

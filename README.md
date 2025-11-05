# Traffic Control System
![streamlit-simulation](https://res.cloudinary.com/din3dx0dt/image/upload/v1762369471/streamlit-snapshot_c8v9vh.png)

## Introduction
Urban traffic management and road safety remain critical challenges in developing smart cities. Conventional traffic light systems rely on *fixed timing cycles*, often failing to adapt to real-time variations in vehicle and pedestrian densities. Simultaneously, *poor road conditions* such as potholes significantly increase accident risks and reduce driving comfort. In this project, we propose an **upgraded version of YOLOv8-based Intelligent Traffic Control System** which is enhanced with **ADAS (Advanced Driver Assistance System)** data for real-time road surface monitoring. 

The system *dynamically adjusts traffic signal durations* based on detected traffic and pedestrian densities while simultaneously determining *safe speed limits using acceleration and gyroscope* data derived from ADAS sensors. This hybrid approach combines camera feed in pictures and sensor analytics to achieve adaptive, safety-oriented traffic control, contributing to more resilient and context-aware urban mobility.

## Past Literature
1. **Smartphone / IMU-based pothole detection (threshold + ML).**  
Many works use *accelerometer + gyroscope* from phones to detect potholes via **RMS/peak detection**, handcrafted features, or lightweight ML (SVM, RF). [(more info)](https://www.researchgate.net/publication/333394788_Smartphone-Based_Pothole_Detection_Utilizing_Artificial_Neural_Networks)
2. **YOLO (real-time CV) for traffic monitoring.**  
**[YOLO](https://docs.ultralytics.com/) (and recent YOLOv8)** is widely used for real-time vehicle/pedestrian detection and counting; many works build traffic-flow estimators, vehicle classification, and trackers (SORT/ByteTrack) on top of YOLO for live signal control.
3. **BATCS**
The Bengaluru traffic police has announced deployment of the *Bengaluru Adaptive Traffic Control System(BATCS)* designed to optimise traffic flow and reduce delays accross cities. [(More info)](https://timesofindia.indiatimes.com/auto/policy-and-industry/bengaluru-police-deploys-ai-based-traffic-control-system-heres-how-it-works/articleshow/114139000.cms)
4. **Roughness to speed relationship (empirical evidence).**  
Econometric work shows road roughness (IRI) correlates with lower vehicular speeds; one study finds **≈11% decrease in average speed per +1 SD increase in IRI**, a useful empirical anchor for mapping roughness → speed reduction. [(Link to paper)](https://www.nber.org/system/files/working_papers/w29176/w29176.pdf)

## Implementation
This project integrates **YOLOv8-based traffic monitoring** with **ADAS (Advanced Driver Assistance System) sensor analytics** to create an intelligent traffic management and safety system. The model dynamically adjusts **traffic light durations** and **speed limits** based on real-time conditions which include vehicle density, pedestrian presence, and road surface irregularities(potholes). By combining vision-based and sensor-based intelligence, it aims to enhance both **traffic efficiency** and **road safety**.
### Counting task: YOLOv8m model [(code)](https://github.com/Soumyajit0803/traffic-control-system/blob/main/YOLOv8_density_tracker.ipynb)
The current implementation uses a **custom wrapper around YOLOv8m model** to count the number of pedestrians and vehicles from camera image feed obtained **every 5 minutes**. 
The custom wrapper **divides the image into a 2×2 grid** and applies the counting algorithm to each of the four cropped regions individually to enhance counting precision.
Currently, the *untrained* YOLOv8m model with custom parameters operated under the setup mentioned above gives an accuracy of around **80%** *(improvement from 67%, without wrapper)* which translates to **miscalculation of only 5-6 pedestrians/vehicles**.

**Proposed Signal timing calculation:**
1. **Defining constants**

	$S_{min} = 30s$, minimum possible light duration

	$S_{max} = 120s$, maximum possible light duration

	$\alpha = 0.8, \beta = 0.5, \gamma = 0.6$, weights

	$\lambda = 0.8$, smoothing factor

	$T_{t-1}$ = Previous signal time

	$D_t$ = Traffic density

	$P_t$ = People density$

	$I_t$ = Road Irregularity Score

	Here, $\alpha$, $\beta$ and $\gamma$ denote the weights for people, traffic and irregularity scores that together determine the signal timing.

2. **Timing Computation**
	
	Following is the calculation of green signal timing. For red signal, we just swap the $D_t$ and $P_t$ values, keeping the formula same.

	$R_x = clip(\alpha*D_t - \beta*P_t + \gamma*I_t, 0, 1)$

	Where

	${clip}(x, 0, 1) =\begin{cases}0, & x < 0 \\[0pt]x, & 0 \le x \le 1 \\[0pt]1, & x > 1\end{cases}$

	To keep the value within $S_{min}$ and $S_{max}$,

	$R_t = S_{min} + (S_{max} - S_{min}) * R_x$

	Finally, to avoid **abrupt fluctuations** we apply **smoothing**. Without it, even small sensor changes in traffic, pedestrians, or road irregularity could cause large, unstable shifts.

	**$T_t = \lambda*T_{t-1} + (1-\lambda)*R_t$**



### Road irregularity tracker: ADAS [(code)](https://github.com/Soumyajit0803/traffic-control-system/blob/main/road_irregularity_tracker.ipynb)

For this we need a basic ADAS sensor that can provide **3-axis data of accelerometer and gyroscope** separately for detecting road irregularity.

**The proposed road irregularity index calculation:**
1. **choosing parameters**

	$acc_z = accelerometerZ$

	$gyro_z = \sqrt{gyroX^2 + gyroY^2 + gyroZ^2}$

	> `gyroX` and `gyroY` would indicate pitching or rolling (e.g., bumps or banking turns). `gyroZ` tracks how sharply the vehicle turns (left/right).

	> Road roughness primarily produces vertical vibrations. Hence only `accelerometerZ` has been considered.

2. **Taking sliding window RMS of accelerometer and gyroscope signals**
	$acc\_z\_rms= sliding\_rms(acc_z)$

	$gyro\_z\_rms= sliding\_rms(gyro_z)$

	RMS measures the energy of the signal. *Taking it over a sliding window helps analyse how it changes over time*.
  
	Window here is a fixed length of consecutive samples. 
	> If the sampling rate is e.g. 50 Hz, there will be 50 samples per window.

3. **Roughness Index (R\)**
	Combine the RMS signals of accelerometer and gyroscope to produce **roughness index R**. A weighted combination is used, giving greater weight to acceleration changes, since acceleration is more affected when encountering a pothole.

	$W_A = 0.7$
	
	$W_G = 0.3$

	$R_{raw} = W_A*acc\_z\_rms + W_G*gyro\_z\_rms$

4. **z-scaling Roughness index ($R_z$)**
	> Raw roughness index depends on **sensor scaling and vehicle type**. A sensor mounted loosely may detect bigger vibrations. A smooth car suspension reduces the same bump's reading. Hence we **normalise** the value for uniformity.
	
	> Z-scaling, or standardization, is a data preprocessing technique that rescales features to have a mean of 0 and a standard deviation of 1
	
	$R_z = \frac{R_{\text{raw}} - \mu_R}{\sigma_R}$
	
	where,
	$\mu_R$ denote mean, and
	$\sigma_R$ denote standard deviation of the signal R.
	
	$R_z$ is our final, time-varying, **normalized signal** showing how roughness changes over time — but now expressed in `standard deviation units`. Interpretation:
	
	- $R_z(t) = 0$ : average road vibration
	- $R_z(t) > 0$ : rougher than average
	- $R_z(t) < 0$ : smoother than average

5. **Roughness score over the segment and calculation of speed limit**
	Once we have the signal representing the roughness over the segment, we need **one representative number** for scoring the segment of road based on roughness. The score directly determines how much the speed limit should be decreased from its maximum value.
	
	For the score, we take the $85^{th}$ percentile value of the signal $R_z(t)$. Using the **$85^{th}$ percentile** keeps the focus on the **rougher end** of the distribution. Taking an average will, instead, smoothen out dangerous spikes.
	
	$R_{score} = np.percentile(R_z,  85)$

	$S_{regulated} = S_{base} * max(f_{min}, 1-k*R_{score})$
	Where 
	* $S_{base}$ = 70 kmph, can be tweaked based on road type
	* $f_{min}$ = 0.3, the lower bound. Speed cannot decrease below 30% of the base.
	* $k$ = 0.11, speed reduction per roughness unit. Based on [this data](https://www.nber.org/system/files/working_papers/w29176/w29176.pdf).

	![sample-roughness-signal](https://res.cloudinary.com/din3dx0dt/image/upload/v1762375802/roughness_signal_xfzv1u.png)
> Sample roughness signal obtained after the above mentioned preprocessing steps. While the blue line represent the signal, orange line  represent the $85^{th}$ percentile limit. Encountering a pothole corresponds to sharp spikes above the orange line

6. **Roughness score as a determinant in traffic control**

	Since the roughness of the road affects vehicle speed, the traffic control needs to be altered for **longer red and green phases** to compensate for lower feasible driving speeds and increased vehicle delay respectively. Additionally, the road will be restricted for trucks/buses/other heavy vehicles that may aggravate the already damaged road condition.

## System requirements
1. **ADAS system**
	The accelerometer and gyroscope signals can be obtained from a typical **Inertial Measurement Unit (IMU)**. A professional-grade **Inertial Navigation System (INS)**, which is a device that combines a high-precision IMU with a GPS, will serve best for the goal.
	
2. **Connectivity:** 
A **4G/5G cellular module** is required for real-time or periodic uploading of sensor data packets to the cloud infrastructure.

3. **Cloud Infrastructure (Data Aggregation & Analysis)**
	The cloud service handles the large-scale aggregation and analysis of ADAS data. It requires:
	- A **high-throughput service** (e.g., IoT message queue) capable of receiving and buffering data from thousands of vehicles simultaneously.
	- **Serverless computing resources** (e.g., AWS Lambda) to **execute the road irregularity analysis** (RMS, z-scoring, 85th percentile) on incoming data
	- A **scalable NoSQL database** to store and serve the **"Road Irregularity Index" (RIS)** and the calculated `S_regulated` for all road segments.

4. **Edge-Computing Subsystem (Intersection Control)**
	This local unit performs real-time traffic analysis, speed limit display, traffic restriction and final signal actuation. it requires:
	- A **low-power Edge AI device** (e.g., NVIDIA Jetson, Google Coral) capable of running the YOLOv8m model. 
	- A **high-resolution (1080p+), outdoor-rated IP camera with night vision** to provide image snapshots for traffic and pedestrian counting
	- A stable internet connection (wired or 4G/5G) to query the cloud database for the latest `RIS` score.
	- A Variable Message Sign (VMS) to display the cloud-derived `S_regulated` speed limit to drivers.


## Workflow
1. YOLOv8 logic
![YOLO-logic](https://res.cloudinary.com/din3dx0dt/image/upload/v1762373485/yolov8-workflow_xy9wdx.png)

2. ADAS logic
![ADAS logic](https://res.cloudinary.com/din3dx0dt/image/upload/v1762369811/adas-workflow_x3qdzw.png)

3. General logic
![general-logic](https://res.cloudinary.com/din3dx0dt/image/upload/v1762373485/minified-traffic-control-system_n0ruju.png)

## Benefits
1. **Car Safety:** Prevents **unexpected jerks or accidents** caused by potholes or uneven roads. Having a regulated speed limit reduces the impact of any approaching pothole or similar irregularity, improving drive experience.
2. **Road Safety:** Regulated speed limit and restricted vehicle access  prevents **aggravating the road condition** by high-speed and heavy vehicles in worn-out roads. 
3. **Optimized analysis**: Using image feed instead of continuous video feed **saves both memory as well as GPU load**. Memory is saved by dealing with images instead of videos, and lesser the data size, quicker the model algorithm runs, requiring much less GPU load. The basic logic behind this alternative is that in real world traffic/people density is not something that changes too frequently. However, the 5 minute period may be tweaked based on feedback algorithm after implementation.
4. **Accuracy independent:** The model performance stats currently on unseen data reveals an accuracy of 80%, with miscounting of 5-6 pedestrians/vehicles -- a number small enough, and most importantly, safe enough to be ignored. Hence, achieving higher accuracy in counting is not that significant-- saving unnecessary training loads.

## Future Scope
1. **Tuning the parameters:** Currently the parameters used are purely based on logic and there is no proof behind it. The control system can be **refined with feedback** from drivers which cab be probably used to train a model.
2. **Upgradation to multiple lanes:** The traffic control can be upgraded to **manage traffic for multiple lanes** crossing each other. New determinants include the calculation of traffic and pedestrian densities along with road condition for other crossing lanes.
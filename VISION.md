# Ito — original vision transcript

Original speech-to-text transcript, preserved with its transcription errors. Product-intent context; subsequent decisions in AGENTS.md and current requirements take precedence.

Okay.

Speaker 1   00:05
I am recording this for a transcript that will go into a prompt. Um, asking for a? Big task. So? I've done a couple of attempts as creating Ito Ito. Um. Ito is immersive Tello operation. That's what it stands for. Software built entirely for piloting robots. Um, those previous attempts.

They were kind of forays experiments into. Five coding. Um, just still learning how it all works. It was made with Dumber models. Uh, and I'm I've been a little disappointed with it. I would, I would kind of shapes up. So, I want to start over. Uh, with maybe Opus55 or?

Gpg Astra. Um.

And I just want to do this big prompt to get the vision. Uh, out of my head. And into the agents that builds the thing. Instead of taking it step by step. And micromanaging everything. So, let me explain. Ito stirred immersive tele operation software. Immersive means VR tele operation means controlling robots.

Or piloting. I want to use the word piloting consistently. Most tele operation software treats the pilot experience as secondary. Uh, it's usually a basic tool for collecting demonstrations and training robots policies. So,

Speaker 2   01:58
A guy records himself picking up an object. In order to train a model that lets the AI pick up the objects.

Speaker 1   02:07
That when you say tele operation, that's what people think of and in the robotics world. Ito is different. It's, it's a totally different approach. Ito is unapologetically and exclusively. Tell operation for the sake of Telo operation for controlling the robot.

That is, its only job. It's gonna do this one thing, and it's gonna do it really well.

Uh, so, yeah, it's sole purpose is to make remotely operating a robot comfortable for the human pilot. I envision future where people Pilots every type of robots from their home or office, or wherever they are. Uh, this could be humanoids, drones. Little cars, anything? This could enable disabled people to act through robots in places their bodies cannot easily take them and allow people to explore or work in environments that might be hostile to humans like disaster areas where you might not want to bring a dog.

But a robot could go there. Um, and if there's human eyes and hands behind it, it could be very flexible.

It always intended to support humanoids, droids, vehicles, meccas, anything.

Any robot form that that doesn't even have to fit a consistent category or existing category. Um. I want to support everything I want to map any kind of human movements. To any kind of robot. And that can be as simple as. Controller buttons. And maybe a camera feed going into the the user screen.

Or at least, you know, to their VR goggles. But maybe as immersive as? Uh, the movement of the pilot's hand. Being translated into some kind of actuator. So that your physical movement really controls the robots in, you know, a very? Human way. As if as if you are the robot.

It translates to Pilot's tracked pose and controller input into control instructions appropriate to the Pilots of robots. That's what I've been saying. In the other direction, it translates the robot's sensor inputs into a comfortable, immersive 3D reconstruction of its surroundings. So that last line? Is kind of where most of my brainstorming has gone into at this point.

Allows back and forth about how it might work. Um. But basically, what I want is more than just a camera feed. Um, most cameras will have one or maybe two cameras, a stereo feed. Exterior cameras, maybe even calibrated. Um. But I don't want to just stream that into the pilot's eyes because I think that's not comfortable.

There's going to be latency. And when you turn your head. That movement has to go to the robot. The robot has to turn its head. The camera has to capture the Turning of the camera. Um. And then it sends that feedback to the pilot and then what you're gonna experience as a pilot is you turn your head and like half a second later, which is way too late.

Uh, you will see. Where your head has turned. That's gonna be disorienting, and it's going to make people sick. Um. So the pitch is. Whatever kind of sensor inputs a robot might have, and the main thing is going to be cameras, 2D cameras, and maybe stereo cameras later. Or maybe right away.

Maybe those two things one stereo, one single camera? Global shutter. And. A stereo camera, Global shutter, and then maybe in the future, we can expand that support rolling shutter supports. Time of light sensors and everything else. Lidar stuff like that? Um. But okay, we want to support those cameras.

We want to stream that. To the pilots. And then. We don't want to show that directly. We're going to feed it into. An algorithm, and that's going to be a kind of Slam algorithm. Uh, there's several AI models. Uh, one is called Master Slim. Mono GS. There's a couple that I might link to in the prompt.

Um. These algorithms can basically take any. Video inputs. Map the position of the camera to a point in physical space relative to the, you know, the first framework or whatever. Um, and then what they can do is turn whatever is seen in in that video feed into a 3D scene.

And that could be a gaussian splat. Or maybe something else, but I I'm kind of. Going for gaussian Splats here. There's models that go directly from camera inputs to gaussian Splats. Uh, and you know, it seems like a really cool technology to to build on top of in in this case.

So you have this gaussian Splat? That's like the last couple of seconds of video footage from the robot. Uh, it turns into this 3D scene of what the robot is seeing right now. But also some pests information about, you know, when the robot trenches head to the right? You can see what's to the right, but you can also.

There's a also a bit of the scene. Um. Of what it captured a second ago, and it's still in that 3D scene. Um. Then, you bring that to the clients. You know, the pilot software? Running in VR? Um, the pilots can look around in this scene. The position of the the basically the camera.

From which the pilot sees. Matches. Um, the robots. Camera in the physical world. If you get what I mean? You're always, you know, in the real worlds, and in this 3D scene, you're always in the same spots as the where the robot is. Um, but technically, you can look anywhere.

So? You might turn your head as a pilot. And at 90 or 120 FPS. Your view updates. So that it's not a dizzying, you know? Puke worthy experience. And then the holes. This whole system of? You sending pose information to the robots, the robot moving its body relative, you know, to match yours.

Capturing new information about the world. Um, and then feeding it into this. Gaussian Splats algorithm, which updates to 3dc in relative to, you know, the most recent video footage. Um, which then updates the pilot. It's like a, it's like a loop. A circle, or like a pipeline that goes on forever.

And maybe it does, it's like each part is like asynchronous from each other. So you know, I might. The pilot might send movements data at, like, 60, hertz. Um. The video stream might be 60 yards, but then the gaussian Splat algorithm might be the only like 10 or 20 Hertz because it's a big, big, slow AI model.

Uh, but then. The kind of game engine that renders this for like a VR goggle. That might run at 90 or 120 hertz, so it looks really smooth.

I hope this is kind of giving you the big picture. Of the idea so? You send post data from a VR goggles and the controllers, and maybe. Body trackers. It's all like VR stuff like VR chat stuff. That goes to the robots. Hi, there's some algorithm or like a policy.

Uh, tries to match that pose look in the same direction as you're looking. Um, sense camera footage back, and you know, audio as well. Microphone, uh? Stream. Audio stream. Um. And then a piece of software. Outside of the robot, probably. Uh, turn set the camera footage into an updated.

Cautions Plus. And the pilots sees the gaussian splats. I think this is the way to go. To make that original pitch of, you know, comfortable and immersive tele operation. To make that a good experience. You know to make you want to pile as a robot because it's fun or useful.

So, a couple of things are going to be needed to build this. It's a big project.

That's gonna have a lot a couple of parts to this. So, first of all, there's the client. Um, this client is going to. Run either on a VR headset Standalone. Natively, I'm not sure about that. Yes.

I might say. No, it's not because you know, the hardware is just weak. Um. What I might want is it's basically going to be. A PC software. That runs in steamvr. You know, try to keep its cross-platform. So, it runs on Windows and Linux. I love Linux, and I want to build for it.

But right now, VR for Linux is not, not good, and Probably not. Reliable enough to build on top of? I need something that works, and then we can make it. Nice.

So, yeah, a VR application. Let's start with Windows. Um, something that can can work with steamvr or or preferably open XR. Um. And then you know if you have a standalone headset like you meta Quest, uh, 3 or pico4, you can use Virtual desktop to connect to your computer to use Ito that way.

I think that's a good first version to to build, and then we can support other platforms later. Um, the reason I want to run it on a computer and maybe not on weaker Standalone Hardware? Is because of that. Gaussians, plus generating AI model like Master Slam. Or mono.js or whatever you need a video card for that.

Um. I might say, let's. Nvidia. Uh, as a kind of main target platform like an in Nvidia GPU in a Windows PC. That's kind of the main target if we can do AMD. Gpus as well, that that's nice. Um, but it's optional. Um. I'm already kind of thinking about performance here, right?

This has to be really fast. Uh, because if it's slow, then it's not a comfortable experience, and that that's the number one thing. Um, so we're running it on on the Windows PC. Uh, where there's a video card so we can run that model and? Basically. Generates that gaussian Splats.

And also render it in the same application. In the same memory. You know, if you have, you know, one of the earlier earlier iterations was? Um. That there's basically three applications, just robots. And then there's a server with the GPU. And then there's a client, maybe like a web client webxr.

Which seems really neat and and cleanly architected. But the problem is, you're gonna have to generate generate that, um. Gaussian splats, and then you're gonna have to stream it to the clients. That's IO, and and that's going to be. Slow, and it might not even be the bottleneck here.

You know, because the the big AI model the like, Master slam is going to be. A big bottleneck.

But I feel like.

Um. If we really think long term. You know? Gpus are going to get faster. AI models aren't going to be able to run a lot faster. It'll become about it. It will no longer be the bottleneck, and maybe if we can find a new ammo or train a new one?

That does this job in a really efficient way? You just never want your kind of core architecture of the application to become the bottleneck. Especially if it's silly i o that could have been prevented, so that's okay so that. That's why, right now, I'm thinking, there's two applications. One that runs on the robot because it's also always kind of a separate computer and.

Um, it won't have strong enough Hardware. And one that is basically. The clients are essentially the pilot software, which also runs. Translating camera footage. Into 3D scene. Which then also renders it. For VR.

Um, let me think what else can I say about this? Okay, so the robot software? Um.

I don't have a robot right now, and that's been kind of a challenge for this project and for, like the first two iterations of this. Because I had to kind of improvise there, and none of the solutions were ideal. My, I first thought I might do like a Raspberry Pi, and I'd buy some cameras, and that's like version 0.1 of like a mock robots that I could build on top of.

Um, but that's not fun. That's not a rewarding process. I, I kind of wanted a real robots. To build this for. Um, and it would be just the first robots to build this, for I want to support every robot in the long term, but I want a real consumer robot.

Uh, at home to work with and to test for against and to use the software with. Otherwise, it feels kind of stupid to build something that you can't even use. In the intended way. Then one day came micro duck. By pollen robotics, and that's a hugging face. Daughter company.

Um, it's an affordable robot, and I instantly ordered it, and I still don't have it yet. Um, but I bought it. Mainly for Ito. Uh, basically, I bought this 400 Euro robots. To build Ito. For it because you know this idea has been in my head for a long time, and I really want to make it.

I needed a robot to build it for. Uh, and I think. The micro duck. Is going to be the first one. Of many, I hope. But it's a real robot. And it's, it's s*****, and it's tiny. It's really, literally, a duck, uh, it has one camera. I think it's Global shutter, which is good, but it's a single camera, so there's not going to be any depth information.

So, it's going to be difficult to. Create a 3D scene, but I have seen. Ai models that can do it. They're going to be slow, like just a couple of FPS. Uh, but functionally it's gonna work. It's going to be able to create a scene. And we're going to be able to build this software.

So, yeah. The micro duck is almost on the way. Um. And we're gonna build. There's a lot of resources, uh, on the internet, about it already. There's a simulator. Uh, we might even be able to simulate a microduct, including its camera footage. I have seen some things like that, but I'm not sure.

We can maybe look into that the test of software before the robot is here. And then we're actually building something for a real robot instead of just a simulator. Um. I think microduck runs Linux. It's, it's gonna be a week, week, week, computer. There's. A neural processing kind of tpus.

In there. But it's not going to be strong enough for anything, except maybe the policies that, for, for its own movement. So, we're not going to rely on this. Uh, we're just gonna. Maybe use their policies for walking and movement or whatever. Uh, but the main thing? Kind of robot side of the software.

It's going to do is record a camera stream. Um. And then, you know, feet movements pose information back into it so that it has to. Uh, copy that.

So, yeah, we're calling that, like a robot driver. Um, and the idea is. There's one Ito application.

Um. But there's going to be many robot drivers. And what this driver is going to be is basically. It has to serve as a kind of adapter. You know, the design patterns adapter pattern? Uh, where Ito speaks one language to every robot, the same kind of commands going back and forth.

Um. And the robot's driver takes those and turns them into robots, specific Instructions. And you know, cuz they'll they'll always have some kind of SDK. Um, they're gonna have different actuators. They're going to be shapes, you know, physically differently. Um. It's, it's basically an adapter that takes the human pose.

Converts it into something that the robot can can do. Um, and then it takes sensor information and sends that back to Ito. Um, the long term is that? We're gonna make the robot drivers. Humanoid robots. Uh, drones like DJI drones. Boats, cars, everything.

And basically the the kind of API or the. The communication protocol between a robot driver and Ito. It's common, it's like shared code. And it should be the same.

Um. Okay, the client. The first two iterations. I was kind of. Interested in doing a web XR clients? And then having like a python server application because you know that the AI models? Um, using python to run AI, like tensor RT and everything. Python is a good language for that.

Uh, but then for, like the VR clients, I was thinking JavaScript web app. Using webxr. And then. 3js and A-frame. Uh, it's. It's a stack I'm familiar with. I've used it before. Um, and there's a 3js. The name I? Uh, kind of forgot about. I can look it up.

One moment.

Oh, I found it. It's called sparkjs. It's a 3D gaussian splitting renderer for 3js. Um. So, you know, imagine? You have this python thing that does the AI part. It sends Splats information to the client, or, you know, it's the client asks for it, maybe over websockets. Um, and then this JavaScript web app, which starts a web XR scene using 3gs and an A-frame.

Um, and then it uses spark JS to render that glossian. Uh, Splat.

And what I found elegant about this is? That most of the work has already been done it. This is all libraries that are really going to help us out and keep our own code base really small and elegant because we don't have to build any of this from scratch anymore.

The VR part is done, you know? The A-frame and 3gs? Are really good application for building VR. Applications. And as barkjs is like our whole Library, that's all we need to do. The gaussian splitting parts?

Um. I still think there. There are some good ideas there, and maybe you know, we, I'm still kind of open to going for it. But as I explained earlier. I feel a kind of performance bottleneck between the clients and the the server. Um. Because, you know, now there's three applications, and they all need to talk to each other.

There's the robots, the server, and the clients, and there's a bunch of lines going between them and a whole bunch of i, o. And I feel like we're gonna hit bottlenecks there. So? Uh. The clients. And the server in one is going to be like a Windows application. Uh, but I'm kind of allergic to using.

Game engines. Like Unity or or unreal? Uh, even if Unity. Is pretty good with.

Speaker 2   27:58
Vr apps and VR games.

Speaker 1   28:02
Um. It doesn't really feel like something an AI agent is gonna excel in because it's Unity. It's just a GUI. There's going to be a lot of clicking around, you know? Maybe maybe you can? Control. Its programmatically completely, but. I don't know. It feels icky. I want to song.

I want to build something that's like pure code. Uh, like, raylip, you know? Uh, so we need to explore our options there, you know? I feel like, I think, relab actually has something VR, but I'm not sure if we can, if that's good enough for us. Uh, Godot. I really like Godot, and they also have some kind of VR support.

Uh, they probably don't have gaussian splitting. That's, that's going to be the hard part if we're going to do a native clients. Gaussian splatting its early days, it's a it's a young technology. And right now. A lot of the work in gaussian splitting is pretty much only in web.

And WebExr. And 3js. Uh, and that's going to be kind of tough for us if we're not going to use all of that. So, because we know we're gonna have to do a lot of work on our own? Um, there might be gaussian Splat renders. For like native game engines?

Or or like Graphics libraries. I don't know about them, though, and I don't know how good they are, and if they're well maintained, and if they're going to be well maintained in the next. 10 years, you know?

Okay. Um. One last thing. Out of scope items. Um. Originally, I had this idea where? Um. You would have basically a fleet of robots. And then, when you open eto the application, you might see, like a grid. Um of all the robots that you have access to, and then maybe there's like a server, a central server that manages this, and then you have a bunch of clients and a bunch of robots.

Uh, and the server is the orchestrator and manager of sessions. Um, and then basically the pilot just clicks on a robot that he wants to Pilots. Starts a session. Uh, claims that robots for themselves and then other Pilots they might see their robots. As you know, already in use?

Um.

As I was developing math idea and working that out, I suddenly lost a lot of motivation. Like, why am I even doing this? You know, there is some coolness to it. But, like, who is this for? Not for me, for sure. All I all I really want to build is comfortable robots by the thing for personal use, mostly.

Uh, not really for, you know, a business running? A fleet of robots that I don't really care about that. It might be useful to someone someone might, you know, pay me to do it to build it. But I'm not motivated to do it all by myself. Uh, and this is ultimately right now.

It's a hobby project that I'm just making for myself. For me to control the micro duck. Um. Or better yet to become the micro duck. Possessing it. Ito is a name that I kind of came to and and settled on, but I spent a really long time. On the word, like possession.

But every single idea related to that word just seems creepy and, and like, eery and. Uh, awful, because you know it. There's a negative connotation with the word possession. But it is kind of accurate you're possessing a robot. Um, so it's it's always a nice name, the kind of corporate friendly.

And I still like the name Ito and and the fact that it stands for what it really is, and that's immersive Style operation. And it's acute name.

Um. Out of scope, so no fleets of of robots Ito is just going to be one pilot one robots always. Um. For version one at least. There's no orchestrator. There's no session management. There's there's none of that because there's never going to be two pilots running. One instance of this software?

There might be multiple robots that might connect to you. And like, I advertised themselves to just to say, hey, you can connect to me. Um. But even that is for the first version. Don't really? Let's not worry about that. Uh, I just want to do one Pilots one robot and get that working really well.

And once it's really good. And it does this one thing really well, you know, minimalism? Um, Unix philosophy. Um, only then do we expand if if that works really well?

Um. So, yeah, that that's the big one that I don't want to do.

I think I'm kind of running out of words here. I've been talking for more than 30 minutes.

I don't know the exact implementation details, and I might kind of let go of that. As a manager of AI agents, I think. I'm just gonna pass this transcripts. Over to? Uh, some kind of orchestrator like Hermes. And then Hermes puts coding agents like codecs or Cloud code CLI to work to build this thing.

And maybe. Um, create a high-level architecture of how this is. You know, be structured. Uh, and then splits this big, big, big idea into. Smaller, actionable issues. That a coding agents can then work on. And then one by one. We we build up this piece of software. And I, I.

Hope is that? When the micro duck is here? Um, I'll have this application, maybe on on Steam, or or just free on the internet. That people can download, and they can Pilot into their robots. And get a first glimpse.

Being. Something else. Stepping out of their own body. And into a robot body. Uh, which? Some people might find weird. Or creepy or anything? But there's also a really empowering elements to it. Because there's a lot of people who can't do a lot of things. And they might be sick or obese or.

Elderly. Uh, but when they. Step into a body of a humanoid robots. They can do anything. They could fly, even if you know if you pilot the Drone or just anything that that can fly. You'll be so under soaring disguise yourself, but just sitting in an airplane but really flying.

Uh, by yourself. And I think that's really cool. So I'll leave it at this. I might post some links to my existing repository. I originally created a GitHub organization called Ito def's i-t-o-d-e-v-s. And then I created the repository called Ito in. Um, but what I think I want to do is basically.

Start over completely Greenfield, clean clean room or whatever. Um, I'll give you the old repository. Yes, as kind of background information. But I really want to start fresh and and do it right this time and do it the way. I, I hope it's going to be.

Um, and I also kind of want my name to be on it. So, just my username on my GitHub account, and then just an ETO project, and if it ever becomes something big, only then I'll make a GitHub organization because. I think for a long time, it'll just be me working on it.

And my little robots.

37 minutes and 40 seconds. Good.

[Setup]
AppName=LiveSPICE
AppVersion=0.13
AppPublisher=Dillon Sharlet
AppPublisherURL="www.livespice.org"
AppSupportURL="https://github.com/dsharlet/LiveSPICE/issues"
DefaultDirName={pf}\LiveSPICE
UninstallDisplayIcon="{app}\LiveSPICE.exe"
UninstallDisplayName=LiveSPICE
DefaultGroupName=LiveSPICE
SetupIconFile="LiveSPICE\LiveSPICE.ico"
Compression=lzma2
SolidCompression=yes
OutputBaseFilename=LiveSPICESetup
OutputDir=Setup

[Components]
Name: "main"; Description: "LiveSPICE"; Types: full compact custom; Flags: fixed
Name: "vst"; Description: "VST Plugin"; Types: full custom

[Files]
Source: "LiveSPICE\bin\Release\LiveSPICE.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "LiveSPICE\bin\Release\*.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "Circuit\Components\*.xml"; DestDir: "{app}\Components"; Flags: onlyifdoesntexist
Source: "Circuit\Components\Readme.txt"; DestDir: "{userdocs}\LiveSPICE\Components"; Flags: onlyifdoesntexist

Source: "LiveSPICEVst\bin\Release\*.dll"; DestDir: "{pf}\Steinberg\VstPlugIns\LiveSPICE"; Flags: ignoreversion; Components: vst

Source: "Tests\Circuits\Active 1stOrder Lowpass RC.schx"; DestDir: "{userdocs}\LiveSPICE\Examples"; Flags: onlyifdoesntexist
Source: "Tests\Circuits\Passive 1stOrder Lowpass RC.schx"; DestDir: "{userdocs}\LiveSPICE\Examples"; Flags: onlyifdoesntexist
Source: "Tests\Circuits\Bridge Rectifier.schx"; DestDir: "{userdocs}\LiveSPICE\Examples"; Flags: onlyifdoesntexist
Source: "Tests\Circuits\Dunlop Cry Baby GCB-95.schx"; DestDir: "{userdocs}\LiveSPICE\Examples"; Flags: onlyifdoesntexist
Source: "Tests\Circuits\Common Cathode Triode Amplifier.schx"; DestDir: "{userdocs}\LiveSPICE\Examples"; Flags: onlyifdoesntexist
Source: "Tests\Circuits\Common Emitter Transistor Amplifier.schx"; DestDir: "{userdocs}\LiveSPICE\Examples"; Flags: onlyifdoesntexist
Source: "Tests\Circuits\Op-Amp Model.schx"; DestDir: "{userdocs}\LiveSPICE\Examples"; Flags: onlyifdoesntexist
Source: "Tests\Circuits\Boss SD1.schx"; DestDir: "{userdocs}\LiveSPICE\Examples"; Flags: onlyifdoesntexist
Source: "Tests\Circuits\Boss SD1 (no buffer).schx"; DestDir: "{userdocs}\LiveSPICE\Examples"; Flags: onlyifdoesntexist
Source: "Tests\Circuits\Ibanez TS9.schx"; DestDir: "{userdocs}\LiveSPICE\Examples"; Flags: onlyifdoesntexist
Source: "Tests\Circuits\Ibanez TS9 (no buffer).schx"; DestDir: "{userdocs}\LiveSPICE\Examples"; Flags: onlyifdoesntexist
Source: "Tests\Circuits\Marshall Blues Breaker.schx"; DestDir: "{userdocs}\LiveSPICE\Examples"; Flags: onlyifdoesntexist
Source: "Tests\Circuits\Bazz Fuss.schx"; DestDir: "{userdocs}\LiveSPICE\Examples"; Flags: onlyifdoesntexist

[Run]
Filename: "{app}\LiveSPICE.exe"; Description: "Run LiveSPICE."; Flags: postinstall nowait

[Icons]
Name: "{group}\LiveSPICE"; Filename: "{app}\LiveSPICE.exe"

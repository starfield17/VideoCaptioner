#define AppName "VideoCaptioner"
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef SourceDir
  #define SourceDir "..\..\dist\VideoCaptioner"
#endif
#ifndef OutputDir
  #define OutputDir "..\..\release"
#endif
#ifndef TargetArch
  #define TargetArch "x64compatible"
#endif
#ifndef AssetArch
  #define AssetArch "x86_64"
#endif
#ifndef NameSuffix
  #define NameSuffix ""
#endif

[Setup]
AppId={{A55CE993-2234-4ABF-9FA4-65E14B58DBD9}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=starfield17
AppPublisherURL=https://github.com/starfield17/VideoCaptioner
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
PrivilegesRequired=lowest
OutputDir={#OutputDir}
OutputBaseFilename=VideoCaptioner-v{#AppVersion}-windows-{#AssetArch}-setup{#NameSuffix}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
SetupIconFile=..\..\build\release-resources\icons\VideoCaptioner.ico
UninstallDisplayIcon={app}\VideoCaptioner.exe
ArchitecturesAllowed={#TargetArch}
ArchitecturesInstallIn64BitMode={#TargetArch}
LicenseFile=..\..\LICENSE

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "chinesesimplified"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\VideoCaptioner.exe"
Name: "{group}\{#AppName} CLI"; Filename: "{cmd}"; Parameters: "/K ""{app}\captioner.exe"" --help"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\VideoCaptioner.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\VideoCaptioner.exe"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent

@echo off
echo ====================================
echo    ONNX Library - YOLO v11 图像处理
echo ====================================
echo.

:menu
echo 请选择运行模式:
echo [1] 图像分类 (Classification)
echo [2] 目标检测 (Detection)  
echo [3] 运行测试 (Test)
echo [4] 构建项目 (Build)
echo [5] 退出 (Exit)
echo.
set /p choice="请输入选择 (1-5): "

if "%choice%"=="1" goto classification
if "%choice%"=="2" goto detection  
if "%choice%"=="3" goto test
if "%choice%"=="4" goto build
if "%choice%"=="5" goto exit
echo 无效选择，请重试...
echo.
goto menu

:classification
echo.
echo 运行图像分类模式...
echo 注意：请确保在 Program.cs 中设置 mode = "classification"
echo.
dotnet run
goto end

:detection
echo.
echo 运行目标检测模式...
echo 注意：请确保在 Program.cs 中设置 mode = "detection"
echo.
dotnet run
goto end

:test
echo.
echo 运行测试模式...
echo.
dotnet run test
goto end

:build
echo.
echo 构建项目...
echo.
dotnet build
echo.
pause
goto menu

:exit
echo 再见!
exit

:end
echo.
echo 程序执行完成！
pause
goto menu

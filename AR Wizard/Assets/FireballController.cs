using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FireballController : MonoBehaviour
{
    public GameObject Hand;
    public Fireball fireball;

    private Vector3 mousePos;
    private float offsetDistance;
    private int activeEffect = 0;

    private void Start()
    {
        offsetDistance = Hand.transform.position.z - Camera.main.transform.position.z;
    }

    void Update()
    {
        if (!Hand.activeInHierarchy)
        {
            Debug.Log("Hand activated");
            Hand.SetActive(true);
        }

        MoveHand();
        if (Input.GetKeyDown(KeyCode.Mouse0))
        {
            ToggleEffect();
        }
    }

    private void ToggleEffect()
    {
        switch (activeEffect)
        {
            case 0:
                {
                    fireball.StartSmoke();
                    activeEffect++;
                    break;
                }
            case 1:
                {
                    fireball.StartFire();
                    activeEffect++;
                    break;
                }
            case 2:
                {
                    fireball.Shoot();
                    activeEffect++;
                    break;
                }
            case 3:
                {
                    fireball.StartExplosion();
                    activeEffect++;
                    break;
                }
            default:
                {
                    activeEffect = 0;
                    break;
                }
        }
    }

    private void MoveHand()
    {
        mousePos = Input.mousePosition;
        mousePos.z += offsetDistance;
        Hand.transform.position = Camera.main.ScreenToWorldPoint(mousePos);
    }
}